#!/usr/bin/env python3
"""
Neural Memory Desktop Layer — native Python+OpenGL 3D graph visualization.
Lightweight: ~30MB vs Chrome's ~500MB. Auto-rotating, pulsing, transparent.

Usage:
    python neural_desktop.py                    # Desktop layer mode
    python neural_desktop.py --window           # Windowed mode (debug)
    python neural_desktop.py --nodes 200        # Limit node count
"""
import argparse
import json
import math
import os
import random
import ssl
import sys
import threading
import time
import urllib.request

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# ═══════════════════════════════════════════
# Config
# ═══════════════════════════════════════════
API_URL = "https://localhost:8443/api/graph"
POLL_INTERVAL = 5.0
NODE_COLORS = {
    'Conversation': (1.0, 0.72, 0.19),   # amber #f5b731
    'Peer':         (0.33, 0.78, 0.47),   # green #55c878
    'Session':      (0.88, 0.25, 0.34),   # red   #e05555
    'Document':     (0.33, 0.72, 0.82),   # cyan  #55b8d0
    'Skill':        (0.66, 0.33, 0.97),   # purp  #a855f7
    'Other':        (0.35, 0.35, 0.42),   # grey  #5a5a6a
}


class NeuralGraph:
    def __init__(self, max_nodes=300):
        self.max_nodes = max_nodes
        self.nodes = []
        self.edges = []
        self.stats = {}
        self.angle = 0.0
        self.rot_speed = 0.002
        self.target_speed = 0.002
        self.pulse_time = 0
        self.last_update = 0

    def load_from_api(self):
        """Fetch graph data from live dashboard API."""
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(API_URL, headers={"User-Agent": "neural-desktop/1.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=5) as resp:
                data = json.loads(resp.read())

            self.stats = data.get("stats", {})

            raw_nodes = data.get("nodes", [])[:self.max_nodes]
            id_set = set(n["id"] for n in raw_nodes)

            # Assign 3D positions using sphere distribution
            self.nodes = []
            for i, n in enumerate(raw_nodes):
                phi = math.acos(1 - 2 * (i + 0.5) / len(raw_nodes))
                theta = math.pi * (1 + math.sqrt(5)) * i
                r = 8.0 + math.sqrt(n.get("total_degree", 1)) * 0.3
                x = r * math.sin(phi) * math.cos(theta)
                y = r * math.sin(phi) * math.sin(theta)
                z = r * math.cos(phi)

                cat = n.get("category", "Other")
                color = NODE_COLORS.get(cat, NODE_COLORS["Other"])
                # Brightness based on degree
                brightness = min(1.0, 0.3 + math.sqrt(n.get("total_degree", 1)) * 0.05)

                self.nodes.append({
                    "id": n["id"],
                    "x": x, "y": y, "z": z,
                    "color": tuple(c * brightness for c in color),
                    "size": max(2.0, math.sqrt(n.get("total_degree", 1)) * 0.8),
                    "category": cat,
                    "degree": n.get("total_degree", 0),
                    "label": n.get("label", "")[:30],
                })

            # Edges between visible nodes
            raw_edges = data.get("edges", [])
            self.edges = []
            for e in raw_edges:
                if e["source"] in id_set and e["target"] in id_set:
                    self.edges.append(e)

            self.last_update = time.time()
            self.pulse_time = time.time()

        except Exception as e:
            pass  # Silent fail, keep existing data

    def get_node_pos(self, node_id):
        for n in self.nodes:
            if n["id"] == node_id:
                return (n["x"], n["y"], n["z"])
        return None


def draw_sphere(x, y, z, radius, color, slices=8, stacks=6):
    """Draw a low-poly sphere (fast)."""
    glPushMatrix()
    glTranslatef(x, y, z)
    glColor4f(color[0], color[1], color[2], 0.9)
    # Use points for very small nodes, quads for larger
    if radius < 1.5:
        glPointSize(radius * 3)
        glBegin(GL_POINTS)
        glVertex3f(0, 0, 0)
        glEnd()
    else:
        quad = gluNewQuadric()
        gluQuadricDrawStyle(quad, GLU_FILL)
        gluSphere(quad, radius * 0.08, slices, stacks)
        gluDeleteQuadric(quad)
    glPopMatrix()


def draw_edge(p1, p2, weight, time_offset=0):
    """Draw an edge line with optional energy pulse."""
    alpha = max(0.03, weight * 0.15)
    glColor4f(0.96, 0.72, 0.19, alpha)
    glLineWidth(max(0.5, weight * 1.5))
    glBegin(GL_LINES)
    glVertex3f(*p1)
    glVertex3f(*p2)
    glEnd()

    # Energy pulse — a bright dot traveling along the edge
    if weight > 0.3:
        t = (time.time() * 0.5 + time_offset) % 1.0
        px = p1[0] + (p2[0] - p1[0]) * t
        py = p1[1] + (p2[1] - p1[1]) * t
        pz = p1[2] + (p2[2] - p1[2]) * t
        glColor4f(1.0, 0.85, 0.3, 0.6)
        glPointSize(2.0)
        glBegin(GL_POINTS)
        glVertex3f(px, py, pz)
        glEnd()


def draw_glow_ring(x, y, z, radius, color, alpha):
    """Draw a glowing ring around a node (for pulse effect)."""
    glColor4f(color[0], color[1], color[2], alpha)
    glLineWidth(1.0)
    glBegin(GL_LINE_LOOP)
    for i in range(24):
        a = 2 * math.pi * i / 24
        glVertex3f(x + math.cos(a) * radius, y + math.sin(a) * radius, z)
    glEnd()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", action="store_true", help="Windowed mode")
    parser.add_argument("--nodes", type=int, default=300, help="Max nodes")
    args = parser.parse_args()

    # Get screen resolution
    pygame.init()
    info = pygame.display.Info()
    W, H = info.current_w, info.current_h

    if args.window:
        screen = pygame.display.set_mode((1280, 720), DOUBLEBUF | OPENGL | RESIZABLE)
    else:
        screen = pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL | NOFRAME)

    pygame.display.set_caption("Neural Memory")

    # Set X11 desktop layer
    if not args.window:
        try:
            import subprocess
            time.sleep(0.5)
            result = subprocess.run(
                ["xdotool", "search", "--name", "Neural Memory"],
                capture_output=True, text=True
            )
            for wid in result.stdout.strip().split("\n"):
                if wid:
                    subprocess.run([
                        "xprop", "-id", wid.strip(),
                        "-f", "_NET_WM_WINDOW_TYPE", "32a",
                        "-set", "_NET_WM_WINDOW_TYPE", "_NET_WM_WINDOW_TYPE_DESKTOP"
                    ], capture_output=True)
                    subprocess.run(["xdotool", "windowlower", wid.strip()], capture_output=True)
        except:
            pass

    # OpenGL setup
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_POINT_SMOOTH)
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
    glClearColor(0.02, 0.02, 0.05, 0.0)  # Very dark, nearly transparent

    # Projection
    glMatrixMode(GL_PROJECTION)
    gluPerspective(50, (W / H if not args.window else 1280/720), 0.1, 200.0)
    glMatrixMode(GL_MODELVIEW)

    # Graph
    graph = NeuralGraph(max_nodes=args.nodes)
    graph.load_from_api()

    # Background poller
    def poll_loop():
        while True:
            time.sleep(POLL_INTERVAL)
            graph.load_from_api()

    threading.Thread(target=poll_loop, daemon=True).start()

    # Main loop
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_q or event.key == K_ESCAPE:
                    running = False
            elif event.type == VIDEORESIZE and args.window:
                glViewport(0, 0, event.w, event.h)
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                gluPerspective(50, event.w / event.h, 0.1, 200.0)
                glMatrixMode(GL_MODELVIEW)

        # Smooth rotation speed transitions
        graph.rot_speed += (graph.target_speed - graph.rot_speed) * 0.02
        graph.angle += graph.rot_speed

        # Pulse decay
        pulse_factor = max(0, 1.0 - (time.time() - graph.pulse_time) * 0.5)

        # Camera
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        cam_r = 25
        cam_x = math.cos(graph.angle) * cam_r
        cam_z = math.sin(graph.angle) * cam_r
        cam_y = 5 + math.sin(graph.angle * 0.5) * 3
        gluLookAt(cam_x, cam_y, cam_z, 0, 0, 0, 0, 1, 0)

        # Draw edges
        node_pos = {n["id"]: (n["x"], n["y"], n["z"]) for n in graph.nodes}
        for e in graph.edges:
            p1 = node_pos.get(e["source"])
            p2 = node_pos.get(e["target"])
            if p1 and p2:
                draw_edge(p1, p2, e.get("weight", 0.1), hash(str(e["source"]) + str(e["target"])) * 0.001)

        # Draw nodes
        for n in graph.nodes:
            # Pulse ring for recently updated nodes
            if pulse_factor > 0 and n["degree"] > 5:
                ring_r = n["size"] * 0.1 + pulse_factor * 0.5
                draw_glow_ring(n["x"], n["y"], n["z"], ring_r, n["color"], pulse_factor * 0.3)

            draw_sphere(n["x"], n["y"], n["z"], n["size"], n["color"])

        # HUD overlay (2D text)
        # We draw this after depth buffer is cleared
        # For simplicity, use pygame's 2D overlay approach
        # (requires rendering to a surface then uploading as texture — skip for perf)

        pygame.display.flip()
        clock.tick(30)  # 30 FPS max for background widget

    pygame.quit()


if __name__ == "__main__":
    main()
