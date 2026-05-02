"""AE-domain bench query corpus — 240 queries across 6 categories.

Per Sprint 2 Phase 7 execution addendum lines 580-1160. Each category gets
40 queries calibrated to a minimum R@5 threshold:

    electrical_contracting   R@5 >= 0.78
    spanish_whatsapp         R@5 >= 0.70
    materials_sku            R@5 >= 0.75
    lennar_lots              R@5 >= 0.80
    financial_calendar       R@5 >= 0.72
    customer_temporal        R@5 >= 0.82

Each query carries:
    - id           : stable identifier (ELC-001 etc.)
    - category     : workflow category
    - query        : retrieval prompt
    - expected_channels : channels likely useful for this query (diagnostic)
    - minimum_rank : ground-truth memory must appear in top-N (default 5)
    - temporal_mode: 'current_or_unspecified' | 'past_window' | 'cross_time'

ground_truth_ids is initially empty per query; fill in after labeling
existing memories via run_ae_domain_bench.py --label-mode.
"""

from __future__ import annotations


CATEGORY_THRESHOLDS = {
    "electrical_contracting": 0.78,
    "spanish_whatsapp":       0.70,
    "materials_sku":          0.75,
    "lennar_lots":            0.80,
    "financial_calendar":     0.72,
    "customer_temporal":      0.82,
}


def _q(qid, category, query, channels, temporal_mode="current_or_unspecified",
       minimum_rank=5, ground_truth_ids=None):
    return {
        "id": qid, "category": category, "query": query,
        "expected_channels": channels,
        "minimum_rank": minimum_rank,
        "temporal_mode": temporal_mode,
        "ground_truth_ids": list(ground_truth_ids or []),
    }


# ---------------------------------------------------------------------------
# Electrical Contracting Jargon (40)
# ---------------------------------------------------------------------------
_ELC = [
    _q("ELC-001", "electrical_contracting", "Which memories mention GFCI requirements for outdoor receptacles?", ["sparse","entity","dense"], ground_truth_ids=[277, 288]),
    _q("ELC-002", "electrical_contracting", "Find the service upgrade notes involving a 200 amp panel.", ["sparse","entity","temporal"]),
    _q("ELC-003", "electrical_contracting", "What did we say about conduit fill on the last commercial job?", ["sparse","temporal","dense"]),
    _q("ELC-004", "electrical_contracting", "Recall the job where AFCI nuisance tripping came up.", ["sparse","dense","graph"]),
    _q("ELC-005", "electrical_contracting", "Which customer asked about a panel replacement and grounding electrode work?", ["entity","sparse","graph"], ground_truth_ids=[5961]),
    _q("ELC-006", "electrical_contracting", "Find notes about trench depth for underground electrical work.", ["sparse","dense"]),
    _q("ELC-007", "electrical_contracting", "What permit issues came up for service mast replacement?", ["sparse","temporal","entity"]),
    _q("ELC-008", "electrical_contracting", "Which project involved EMT conduit and exposed garage wiring?", ["sparse","entity","graph"]),
    _q("ELC-009", "electrical_contracting", "Find memories about load calculation for a service upgrade.", ["sparse","dense","procedural"]),
    _q("ELC-010", "electrical_contracting", "What did we decide about replacing knob-and-tube wiring?", ["sparse","temporal","dense"]),
    _q("ELC-011", "electrical_contracting", "Which memories mention bonding bushings?", ["sparse","dense"], ground_truth_ids=[5961]),
    _q("ELC-012", "electrical_contracting", "Find the customer conversation about recessed lighting spacing.", ["dense","entity","temporal"]),
    _q("ELC-013", "electrical_contracting", "What notes do we have about EV charger installation?", ["sparse","entity","procedural"], ground_truth_ids=[158]),
    _q("ELC-014", "electrical_contracting", "Which job had a failed inspection due to missing labels?", ["sparse","temporal","graph"], ground_truth_ids=[274, 286]),
    _q("ELC-015", "electrical_contracting", "Find the memory about meter socket replacement.", ["sparse","entity"]),
    _q("ELC-016", "electrical_contracting", "What did we record about a subpanel neutral-ground bond issue?", ["sparse","dense","graph"], ground_truth_ids=[5961]),
    _q("ELC-017", "electrical_contracting", "Which customer wanted a generator interlock?", ["sparse","entity"]),
    _q("ELC-018", "electrical_contracting", "Find notes about weatherhead or riser damage.", ["sparse","dense"]),
    _q("ELC-019", "electrical_contracting", "What was the plan for adding dedicated circuits in the kitchen?", ["entity","procedural","temporal"]),
    _q("ELC-020", "electrical_contracting", "Which job involved low voltage wiring coordination?", ["sparse","entity","dense"]),
    _q("ELC-021", "electrical_contracting", "Find memories about failed breaker replacement or bad breaker diagnosis.", ["sparse","dense"]),
    _q("ELC-022", "electrical_contracting", "What did we learn about Chicago electrical permit timing?", ["entity","temporal","sparse"]),
    _q("ELC-023", "electrical_contracting", "Which project used MC cable instead of conduit?", ["sparse","entity"]),
    _q("ELC-024", "electrical_contracting", "Find memories about aluminum wiring remediation.", ["sparse","dense"]),
    _q("ELC-025", "electrical_contracting", "What did we say about smoke detector circuit requirements?", ["sparse","procedural"]),
    _q("ELC-026", "electrical_contracting", "Which job had a short circuit caused by damaged romex?", ["sparse","causal","temporal"]),
    _q("ELC-027", "electrical_contracting", "Find panel schedule or labeling notes.", ["sparse","dense"], ground_truth_ids=[274, 286]),
    _q("ELC-028", "electrical_contracting", "What was the quote context for a detached garage subpanel?", ["entity","sparse","temporal"]),
    _q("ELC-029", "electrical_contracting", "Which memory mentions emergency electrical repair after storm damage?", ["sparse","temporal","entity"]),
    _q("ELC-030", "electrical_contracting", "Find memories about replacing cloth wiring.", ["sparse","dense"]),
    _q("ELC-031", "electrical_contracting", "What did we note about appliance dedicated circuits?", ["sparse","procedural"]),
    _q("ELC-032", "electrical_contracting", "Which customer discussed adding bathroom exhaust fan wiring?", ["entity","dense","sparse"]),
    _q("ELC-033", "electrical_contracting", "Find work notes about commercial lighting retrofit.", ["entity","sparse","dense"]),
    _q("ELC-034", "electrical_contracting", "What project mentioned pulling THHN through conduit?", ["sparse","entity"]),
    _q("ELC-035", "electrical_contracting", "Which notes discuss surge protection at the panel?", ["sparse","dense"]),
    _q("ELC-036", "electrical_contracting", "Find the memory about two-prong outlet replacement.", ["sparse","dense"]),
    _q("ELC-037", "electrical_contracting", "What was the decision on dedicated microwave circuit?", ["sparse","temporal","procedural"]),
    _q("ELC-038", "electrical_contracting", "Which customer asked for exterior lighting on a timer?", ["entity","sparse","temporal"]),
    _q("ELC-039", "electrical_contracting", "Find memories about open neutral diagnosis.", ["sparse","causal","dense"]),
    _q("ELC-040", "electrical_contracting", "What was the plan for permit inspection rework?", ["temporal","procedural","sparse"]),
]


# ---------------------------------------------------------------------------
# Spanish WhatsApp Crew Communications (40)
# ---------------------------------------------------------------------------
_SPA = [
    _q("SPA-001", "spanish_whatsapp", "Busca el mensaje donde el crew dijo que faltaba conduit.", ["sparse","multilingual_dense","entity"]),
    _q("SPA-002", "spanish_whatsapp", "Que dijo el equipo sobre llegar tarde al job site?", ["temporal","multilingual_dense"]),
    _q("SPA-003", "spanish_whatsapp", "Encuentra la conversacion sobre material que no llego.", ["sparse","entity","temporal"], ground_truth_ids=[6671, 6674, 6700, 7363, 7365, 7377, 10188, 10190, 10202, 12681, 12683, 12695, 13715, 13716, 15078, 15080, 15092]),
    _q("SPA-004", "spanish_whatsapp", "Cual fue la instruccion en Espanol para apagar el breaker?", ["procedural","sparse","multilingual_dense"]),
    _q("SPA-005", "spanish_whatsapp", "Busca notas de WhatsApp sobre permiso de trabajo.", ["sparse","temporal"]),
    _q("SPA-006", "spanish_whatsapp", "Que crew member pidio mas cable?", ["entity","sparse"]),
    _q("SPA-007", "spanish_whatsapp", "Encuentra mensaje sobre panel en mal estado.", ["sparse","entity"]),
    _q("SPA-008", "spanish_whatsapp", "Donde hablamos de terminar manana?", ["temporal","multilingual_dense"]),
    _q("SPA-009", "spanish_whatsapp", "Busca mensaje sobre inspeccion fallida.", ["sparse","temporal","entity"]),
    _q("SPA-010", "spanish_whatsapp", "Que se dijo sobre comprar breakers?", ["materials","sparse","temporal"], ground_truth_ids=[6700, 7377, 10202, 12695, 15092]),
    _q("SPA-011", "spanish_whatsapp", "Encuentra notas sobre escalera o ladder en el sitio.", ["sparse","multilingual_dense"]),
    _q("SPA-012", "spanish_whatsapp", "Quien dijo que el cliente no estaba en casa?", ["entity","temporal"]),
    _q("SPA-013", "spanish_whatsapp", "Busca mensaje sobre cerrar la electricidad.", ["procedural","dense"]),
    _q("SPA-014", "spanish_whatsapp", "Que dijo el crew sobre la caja electrica?", ["entity","sparse","dense"]),
    _q("SPA-015", "spanish_whatsapp", "Encuentra la conversacion de cortar drywall.", ["sparse","dense"]),
    _q("SPA-016", "spanish_whatsapp", "Busca mensaje donde dijeron que llegaron los materiales.", ["materials","temporal"]),
    _q("SPA-017", "spanish_whatsapp", "Que nota tenemos sobre permiso en Chicago?", ["entity","sparse","temporal"]),
    _q("SPA-018", "spanish_whatsapp", "Encuentra mensajes sobre almuerzo o pausa del crew.", ["temporal","dense"]),
    _q("SPA-019", "spanish_whatsapp", "Quien dijo que faltaba el inspector?", ["entity","sparse","temporal"]),
    _q("SPA-020", "spanish_whatsapp", "Busca instrucciones en Espanol para etiquetar el panel.", ["procedural","sparse"]),
    _q("SPA-021", "spanish_whatsapp", "Que mensaje menciona conduit de media pulgada?", ["sparse","materials"]),
    _q("SPA-022", "spanish_whatsapp", "Encuentra mensaje sobre problemas con estacionamiento.", ["dense","temporal"]),
    _q("SPA-023", "spanish_whatsapp", "Que dijo el equipo sobre regresar al dia siguiente?", ["temporal","dense"]),
    _q("SPA-024", "spanish_whatsapp", "Busca el mensaje sobre cambiar un outlet quemado.", ["sparse","dense"]),
    _q("SPA-025", "spanish_whatsapp", "Quien pidio la direccion del trabajo?", ["entity","temporal"]),
    _q("SPA-026", "spanish_whatsapp", "Encuentra mensaje sobre la tienda de materiales.", ["materials","entity","temporal"]),
    _q("SPA-027", "spanish_whatsapp", "Que dijo el crew sobre hacer fotos antes y despues?", ["procedural","dense"]),
    _q("SPA-028", "spanish_whatsapp", "Busca notas donde se menciona GFCI en Espanol.", ["sparse","multilingual_dense"], ground_truth_ids=[277, 288, 4666]),
    _q("SPA-029", "spanish_whatsapp", "Que mensaje dice que el cliente aprobo el precio?", ["entity","temporal","sparse"]),
    _q("SPA-030", "spanish_whatsapp", "Encuentra la instruccion de limpiar antes de salir.", ["procedural","dense"]),
    _q("SPA-031", "spanish_whatsapp", "Busca mensaje sobre job cancelado.", ["temporal","entity"]),
    _q("SPA-032", "spanish_whatsapp", "Quien confirmo que ya termino la instalacion?", ["entity","temporal"]),
    _q("SPA-033", "spanish_whatsapp", "Encuentra mensaje sobre falta de energia.", ["sparse","dense"]),
    _q("SPA-034", "spanish_whatsapp", "Que se dijo sobre recoger permit paperwork?", ["sparse","temporal"]),
    _q("SPA-035", "spanish_whatsapp", "Busca mensaje de WhatsApp sobre cable numero doce.", ["sparse","materials"], ground_truth_ids=[6674, 7365, 10190, 12683, 13716, 15080]),
    _q("SPA-036", "spanish_whatsapp", "Quien dijo que el breaker no encajaba?", ["entity","materials"]),
    _q("SPA-037", "spanish_whatsapp", "Encuentra notas en Espanol sobre trabajar en el sotano.", ["entity","dense"]),
    _q("SPA-038", "spanish_whatsapp", "Que mensaje menciona inspeccion el viernes?", ["temporal","sparse"]),
    _q("SPA-039", "spanish_whatsapp", "Busca la conversacion donde se pidieron fotos del panel.", ["entity","procedural","dense"]),
    _q("SPA-040", "spanish_whatsapp", "Que dijo el crew sobre dejar el job clean?", ["procedural","multilingual_dense"]),
]


# ---------------------------------------------------------------------------
# Materials and SKU Resolution (40)
# ---------------------------------------------------------------------------
_MAT = [
    _q("MAT-001", "materials_sku", "Find memories mentioning a Square D 20 amp breaker.", ["sparse","materials_entity"]),
    _q("MAT-002", "materials_sku", "Which job needed Siemens breakers?", ["sparse","entity"]),
    _q("MAT-003", "materials_sku", "Find notes about 12/2 romex purchase.", ["sparse","materials"]),
    _q("MAT-004", "materials_sku", "What materials were needed for the panel upgrade?", ["entity","temporal","materials"]),
    _q("MAT-005", "materials_sku", "Which memory mentions EMT connectors?", ["sparse","materials"]),
    _q("MAT-006", "materials_sku", "Find SKU context for GFCI outlets.", ["sparse","materials"], ground_truth_ids=[4666]),
    _q("MAT-007", "materials_sku", "Which job required a meter socket?", ["sparse","entity","materials"]),
    _q("MAT-008", "materials_sku", "Find memories about wire nuts or connectors.", ["sparse","materials"]),
    _q("MAT-009", "materials_sku", "What did we buy for exterior lighting?", ["entity","temporal","materials"]),
    _q("MAT-010", "materials_sku", "Which material list included THHN black white green?", ["sparse","materials"]),
    _q("MAT-011", "materials_sku", "Find notes about half inch conduit.", ["sparse","materials"]),
    _q("MAT-012", "materials_sku", "Which customer needed a surge protector?", ["entity","sparse","materials"]),
    _q("MAT-013", "materials_sku", "Find memory about load center brand.", ["sparse","materials"]),
    _q("MAT-014", "materials_sku", "What project needed weatherproof boxes?", ["sparse","entity","materials"]),
    _q("MAT-015", "materials_sku", "Find the conversation about buying breakers at Home Depot.", ["materials","entity","temporal"], ground_truth_ids=[4917, 6267]),
    _q("MAT-016", "materials_sku", "Which job required a grounding rod?", ["sparse","entity"], ground_truth_ids=[5961, 11009, 11011, 11682, 13473, 13482, 13485, 13526, 13556, 13570]),
    _q("MAT-017", "materials_sku", "Find material notes about PVC conduit.", ["sparse","materials"]),
    _q("MAT-018", "materials_sku", "What SKU or item did we use for dimmer switches?", ["sparse","materials"]),
    _q("MAT-019", "materials_sku", "Which quote included recessed cans?", ["sparse","entity","materials"]),
    _q("MAT-020", "materials_sku", "Find memory about AFCI breaker part.", ["sparse","materials"]),
    _q("MAT-021", "materials_sku", "What materials were missing on the crew message?", ["materials","spanish","temporal"], ground_truth_ids=[6671, 6674, 6700, 7363, 7365, 7377, 10188, 10190, 10202, 12681, 12683, 12695, 13715, 13716, 15078, 15080, 15092]),
    _q("MAT-022", "materials_sku", "Find notes about a disconnect switch.", ["sparse","materials"]),
    _q("MAT-023", "materials_sku", "Which project needed flex conduit?", ["sparse","entity"], ground_truth_ids=[5955]),
    _q("MAT-024", "materials_sku", "Find materials for kitchen dedicated circuits.", ["procedural","materials","entity"]),
    _q("MAT-025", "materials_sku", "What did we record about panel labels?", ["sparse","materials"], ground_truth_ids=[274, 286]),
    _q("MAT-026", "materials_sku", "Find memory about junction box size.", ["sparse","materials"], ground_truth_ids=[5947, 5966]),
    _q("MAT-027", "materials_sku", "Which job required low voltage cable?", ["sparse","entity"]),
    _q("MAT-028", "materials_sku", "Find notes about LED fixture model.", ["sparse","materials"]),
    _q("MAT-029", "materials_sku", "What material was used for underground feed?", ["materials","dense","entity"]),
    _q("MAT-030", "materials_sku", "Find memory about GFCI breaker versus outlet choice.", ["sparse","procedural","materials"], ground_truth_ids=[4666, 277, 288]),
    _q("MAT-031", "materials_sku", "Which job involved replacing a burnt outlet?", ["sparse","entity","temporal"]),
    _q("MAT-032", "materials_sku", "Find notes about smoke detector model.", ["sparse","materials"]),
    _q("MAT-033", "materials_sku", "What did we buy for the garage subpanel?", ["materials","entity","temporal"]),
    _q("MAT-034", "materials_sku", "Find memory about anti-short bushings.", ["sparse","materials"]),
    _q("MAT-035", "materials_sku", "Which job mentioned wire staples?", ["sparse","materials"]),
    _q("MAT-036", "materials_sku", "Find material list for basement rough-in.", ["materials","entity"]),
    _q("MAT-037", "materials_sku", "What memory mentions permit stickers or labels?", ["sparse","dense"], ground_truth_ids=[274, 286, 13692, 13693]),
    _q("MAT-038", "materials_sku", "Find SKU notes for exterior in-use covers.", ["sparse","materials"]),
    _q("MAT-039", "materials_sku", "Which material was substituted because the first item was out of stock?", ["temporal","materials","graph"]),
    _q("MAT-040", "materials_sku", "Find the memory about returning unused breakers.", ["materials","temporal"]),
]


# ---------------------------------------------------------------------------
# Lennar Lot Lookups (40)
# ---------------------------------------------------------------------------
_LOT = [
    _q("LOT-001", "lennar_lots", "Find all memories for Lennar lot 12.", ["entity","sparse","temporal"], ground_truth_ids=[5179, 4914]),
    _q("LOT-002", "lennar_lots", "What happened at lot 27 before the inspection?", ["entity","temporal","graph"], ground_truth_ids=[266, 281]),
    _q("LOT-003", "lennar_lots", "Which lot had the missing permit paperwork?", ["entity","sparse"], ground_truth_ids=[13692, 13693, 13687, 13688]),
    _q("LOT-004", "lennar_lots", "Find notes about Lennar closeout punch list.", ["sparse","entity"]),
    _q("LOT-005", "lennar_lots", "Which lot needed panel labels corrected?", ["entity","sparse"], ground_truth_ids=[274, 286]),
    _q("LOT-006", "lennar_lots", "Find memory about Sarah's Lennar contact update.", ["entity","temporal"], ground_truth_ids=[264, 268, 280, 282]),
    _q("LOT-007", "lennar_lots", "What was the last action item for lot 44?", ["entity","temporal","procedural"]),
    _q("LOT-008", "lennar_lots", "Which lot had delayed material delivery?", ["entity","materials","temporal"]),
    _q("LOT-009", "lennar_lots", "Find notes for lot address with basement rough-in.", ["entity","dense"]),
    _q("LOT-010", "lennar_lots", "Which Lennar lot mentioned failed inspection?", ["entity","sparse","temporal"]),
    _q("LOT-011", "lennar_lots", "Find all lot notes from last week.", ["temporal","entity"]),
    _q("LOT-012", "lennar_lots", "Which lot required exterior lighting rework?", ["entity","sparse"]),
    _q("LOT-013", "lennar_lots", "What did the crew say about lot 33?", ["entity","spanish","temporal"]),
    _q("LOT-014", "lennar_lots", "Find notes about Lennar builder contact before Sarah.", ["entity","temporal"]),
    _q("LOT-015", "lennar_lots", "Which lot had a missing GFCI?", ["entity","sparse"], ground_truth_ids=[277, 288]),
    _q("LOT-016", "lennar_lots", "Find work order memory for Lennar electrical trim.", ["entity","sparse"]),
    _q("LOT-017", "lennar_lots", "Which lot had attic access issue?", ["entity","dense"]),
    _q("LOT-018", "lennar_lots", "Find notes about lot 18 materials.", ["entity","materials"]),
    _q("LOT-019", "lennar_lots", "What was the quote status for lot 51?", ["entity","temporal","financial"]),
    _q("LOT-020", "lennar_lots", "Which Lennar lot had customer walkthrough notes?", ["entity","temporal"]),
    _q("LOT-021", "lennar_lots", "Find memory about garage subpanel in a Lennar lot.", ["entity","sparse"]),
    _q("LOT-022", "lennar_lots", "Which lot needed smoke detector corrections?", ["entity","sparse"]),
    _q("LOT-023", "lennar_lots", "Find latest Lennar contact preference.", ["entity","temporal"], ground_truth_ids=[268, 282]),
    _q("LOT-024", "lennar_lots", "Which lot was rescheduled because of rain?", ["entity","temporal"]),
    _q("LOT-025", "lennar_lots", "Find notes about lot 40 inspection window.", ["entity","temporal"]),
    _q("LOT-026", "lennar_lots", "Which lot had change order discussion?", ["entity","financial","temporal"]),
    _q("LOT-027", "lennar_lots", "Find all memories connected to Lennar superintendent.", ["entity","graph"]),
    _q("LOT-028", "lennar_lots", "Which lot had panel upgrade discussion?", ["entity","sparse"]),
    _q("LOT-029", "lennar_lots", "Find note where Lennar asked for pictures.", ["entity","procedural","temporal"]),
    _q("LOT-030", "lennar_lots", "Which lot had meter issue?", ["entity","sparse"]),
    _q("LOT-031", "lennar_lots", "Find memory about replacing exterior outlet at a Lennar lot.", ["entity","sparse"]),
    _q("LOT-032", "lennar_lots", "Which lot had punch-list item already completed?", ["entity","temporal"]),
    _q("LOT-033", "lennar_lots", "Find lot notes mentioning Friday inspection.", ["entity","temporal"]),
    _q("LOT-034", "lennar_lots", "Which lot had wrong breaker type?", ["entity","materials"]),
    _q("LOT-035", "lennar_lots", "Find Lennar lot note involving permit office.", ["entity","sparse","temporal"], ground_truth_ids=[4914, 5179]),
    _q("LOT-036", "lennar_lots", "Which lot needed crew to return tomorrow?", ["entity","spanish","temporal"]),
    _q("LOT-037", "lennar_lots", "Find latest status for lot 29.", ["entity","temporal"]),
    _q("LOT-038", "lennar_lots", "Which lot had unresolved electrical issue?", ["entity","graph","temporal"]),
    _q("LOT-039", "lennar_lots", "Find notes connecting Lennar lot to invoice status.", ["entity","financial"], ground_truth_ids=[5179, 4914, 5180, 5531]),
    _q("LOT-040", "lennar_lots", "Which lot had the customer signoff?", ["entity","temporal"]),
]


# ---------------------------------------------------------------------------
# Financial Calendar and Job-Cost Reasoning (40)
# ---------------------------------------------------------------------------
_FIN = [
    _q("FIN-001", "financial_calendar", "Which jobs have unpaid invoices this week?", ["financial","temporal","entity"]),
    _q("FIN-002", "financial_calendar", "Find memories about job-cost overruns for materials.", ["financial","materials","temporal"], ground_truth_ids=[2628, 2659]),
    _q("FIN-003", "financial_calendar", "Which quote was approved but not scheduled?", ["financial","temporal","entity"]),
    _q("FIN-004", "financial_calendar", "What was the payment status before the latest update?", ["temporal","financial"]),
    _q("FIN-005", "financial_calendar", "Find notes about QuickBooks refresh or QBO token issue.", ["sparse","temporal","procedural"]),
    _q("FIN-006", "financial_calendar", "Which customer delayed payment because of PO approval?", ["entity","temporal","causal"]),
    _q("FIN-007", "financial_calendar", "Find job-cost reasoning for the panel upgrade.", ["financial","materials","procedural"]),
    _q("FIN-008", "financial_calendar", "Which invoice needs follow-up tomorrow?", ["financial","temporal"]),
    _q("FIN-009", "financial_calendar", "Find memories about deposit paid before scheduling.", ["financial","temporal"]),
    _q("FIN-010", "financial_calendar", "Which job had labor cost higher than expected?", ["financial","temporal","dense"]),
    _q("FIN-011", "financial_calendar", "Find notes about permit fee reimbursement.", ["financial","sparse"]),
    _q("FIN-012", "financial_calendar", "Which materials were billed to the wrong job?", ["financial","materials","causal"]),
    _q("FIN-013", "financial_calendar", "Find the last calendar reminder for invoice follow-up.", ["temporal","financial"]),
    _q("FIN-014", "financial_calendar", "What did we decide about charging for reinspection?", ["financial","procedural","temporal"]),
    _q("FIN-015", "financial_calendar", "Which customer asked for itemized pricing?", ["entity","financial","temporal"]),
    _q("FIN-016", "financial_calendar", "Find memories about payroll timing for crew.", ["financial","temporal"]),
    _q("FIN-017", "financial_calendar", "Which project has quote sent but no approval?", ["financial","temporal"]),
    _q("FIN-018", "financial_calendar", "Find cost notes for EV charger installation.", ["financial","materials","entity"], ground_truth_ids=[158]),
    _q("FIN-019", "financial_calendar", "Which job needs final invoice after inspection?", ["financial","temporal","entity"]),
    _q("FIN-020", "financial_calendar", "Find memories about warranty work not billable.", ["financial","procedural"]),
    _q("FIN-021", "financial_calendar", "What was the financial calendar action for Friday?", ["temporal","financial"]),
    _q("FIN-022", "financial_calendar", "Which job had supplier receipt missing?", ["financial","materials"]),
    _q("FIN-023", "financial_calendar", "Find memories about sales tax or material markup.", ["financial","procedural"]),
    _q("FIN-024", "financial_calendar", "Which customer paid cash versus card?", ["financial","entity"]),
    _q("FIN-025", "financial_calendar", "Find notes about quote revision after scope changed.", ["financial","temporal","causal"]),
    _q("FIN-026", "financial_calendar", "Which unpaid invoice relates to Lennar?", ["financial","entity"], ground_truth_ids=[4596]),
    _q("FIN-027", "financial_calendar", "Find memories about accounting closeout for April.", ["financial","temporal"]),
    _q("FIN-028", "financial_calendar", "Which job is profitable after material costs?", ["financial","materials","graph"]),
    _q("FIN-029", "financial_calendar", "Find notes about resending an invoice.", ["financial","temporal"]),
    _q("FIN-030", "financial_calendar", "Which customer needed W9 or vendor paperwork?", ["financial","entity","sparse"]),
    _q("FIN-031", "financial_calendar", "Find memories about deposit balance due.", ["financial","sparse"]),
    _q("FIN-032", "financial_calendar", "Which job had change order not invoiced?", ["financial","temporal","causal"]),
    _q("FIN-033", "financial_calendar", "Find calendar memory about QBO reauth.", ["temporal","procedural","sparse"]),
    _q("FIN-034", "financial_calendar", "Which quote had permit fee excluded?", ["financial","sparse"]),
    _q("FIN-035", "financial_calendar", "Find notes about refund or credit.", ["financial","temporal"]),
    _q("FIN-036", "financial_calendar", "Which job had materials bought before customer approval?", ["financial","materials","temporal"]),
    _q("FIN-037", "financial_calendar", "Find memories about insurance or certificate paperwork.", ["financial","sparse"]),
    _q("FIN-038", "financial_calendar", "Which invoices need aging report attention?", ["financial","temporal"]),
    _q("FIN-039", "financial_calendar", "Find memory about job-cost split between labor and materials.", ["financial","materials"], ground_truth_ids=[5875, 2785]),
    _q("FIN-040", "financial_calendar", "Which customer had payment terms changed?", ["financial","temporal","entity"]),
]


# ---------------------------------------------------------------------------
# Customer/Contact Temporal Updates (40) — bi-temporal-heavy category
# ---------------------------------------------------------------------------
_TMP = [
    _q("TMP-001", "customer_temporal", "Who was the main contact before Sarah?", ["entity","temporal","bitemporal"], "past_window"),
    _q("TMP-002", "customer_temporal", "What did we believe before the correction?", ["temporal","contradiction","graph"], "past_window"),
    _q("TMP-003", "customer_temporal", "Which memory was invalidated by the newer customer update?", ["bitemporal","contradiction"], "cross_time"),
    _q("TMP-004", "customer_temporal", "Find the old address before the customer moved.", ["entity","temporal"], "past_window"),
    _q("TMP-005", "customer_temporal", "What was the previous inspection date before it changed?", ["temporal","entity"], "past_window"),
    _q("TMP-006", "customer_temporal", "Who handled the job before Miguel?", ["entity","temporal"], "past_window"),
    _q("TMP-007", "customer_temporal", "What was the material plan before substitution?", ["materials","temporal","causal"], "past_window"),
    _q("TMP-008", "customer_temporal", "Which quote version came before the approved one?", ["financial","temporal"], "past_window"),
    _q("TMP-009", "customer_temporal", "Find notes from before the permit was approved.", ["temporal","entity"], "past_window"),
    _q("TMP-010", "customer_temporal", "What did the crew say before the final correction?", ["spanish","temporal"], "past_window"),
    _q("TMP-011", "customer_temporal", "Which customer contact was valid in March?", ["entity","bitemporal"], "past_window"),
    _q("TMP-012", "customer_temporal", "What happened after the failed inspection?", ["temporal","graph"]),
    _q("TMP-013", "customer_temporal", "Which note superseded the old scope?", ["temporal","contradiction"], "cross_time"),
    _q("TMP-014", "customer_temporal", "Find the earliest memory about this job.", ["entity","temporal"], "past_window"),
    _q("TMP-015", "customer_temporal", "What was the latest valid status?", ["temporal","entity"]),
    _q("TMP-016", "customer_temporal", "Which memory says this job was cancelled?", ["temporal","entity"]),
    _q("TMP-017", "customer_temporal", "What was the status before payment was received?", ["financial","temporal"], "past_window"),
    _q("TMP-018", "customer_temporal", "Find old contact info that should not be treated as current.", ["entity","bitemporal"], "past_window"),
    _q("TMP-019", "customer_temporal", "Which memories are stale but still connected to this customer?", ["temporal","graph","salience"]),
    _q("TMP-020", "customer_temporal", "What was known at the time of the quote?", ["temporal","financial"], "past_window"),
    _q("TMP-021", "customer_temporal", "Which prior instruction was replaced?", ["procedural","temporal","contradiction"], "cross_time"),
    _q("TMP-022", "customer_temporal", "What was the job schedule before rescheduling?", ["temporal","entity"], "past_window"),
    _q("TMP-023", "customer_temporal", "Find the memory valid during the first inspection window.", ["bitemporal","entity"], "past_window"),
    _q("TMP-024", "customer_temporal", "Which customer preference changed over time?", ["entity","temporal"], "cross_time"),
    _q("TMP-025", "customer_temporal", "What did we know before the customer approved the change order?", ["temporal","financial"], "past_window"),
    _q("TMP-026", "customer_temporal", "Find the previous lot contact before Sarah took over.", ["entity","bitemporal"], "past_window"),
    _q("TMP-027", "customer_temporal", "What was the crew assignment before today's update?", ["entity","temporal"], "past_window"),
    _q("TMP-028", "customer_temporal", "Which memory should win when two statuses conflict?", ["temporal","confidence","contradiction"]),
    _q("TMP-029", "customer_temporal", "What was the old materials list?", ["materials","temporal"], "past_window"),
    _q("TMP-030", "customer_temporal", "Find notes created after the last dream cycle.", ["temporal","benchmark_trace"]),
    _q("TMP-031", "customer_temporal", "Which fact was true only until April?", ["bitemporal","temporal"], "past_window"),
    _q("TMP-032", "customer_temporal", "What was valid before the transaction time changed?", ["bitemporal"], "past_window"),
    _q("TMP-033", "customer_temporal", "Find the update that changed customer contact.", ["entity","temporal","graph"]),
    _q("TMP-034", "customer_temporal", "Which memory is newest but lower confidence?", ["temporal","confidence"]),
    _q("TMP-035", "customer_temporal", "What earlier memory contradicts the latest status?", ["contradiction","temporal"], "cross_time"),
    _q("TMP-036", "customer_temporal", "Find all updates between scheduling and inspection.", ["temporal","entity"]),
    _q("TMP-037", "customer_temporal", "Which note was reinforced most recently?", ["salience","temporal"]),
    _q("TMP-038", "customer_temporal", "What was the previous procedural instruction?", ["procedural","temporal"], "past_window"),
    _q("TMP-039", "customer_temporal", "Which memory explains why the date moved?", ["temporal","causal"]),
    _q("TMP-040", "customer_temporal", "Find the old customer preference and the new one.", ["entity","temporal","contradiction"], "cross_time"),
]


ALL_QUERIES = _ELC + _SPA + _MAT + _LOT + _FIN + _TMP


def get_queries(category: str | None = None) -> list[dict]:
    """Return all queries, or just queries for one category."""
    if category is None:
        return list(ALL_QUERIES)
    return [q for q in ALL_QUERIES if q["category"] == category]


def category_counts() -> dict[str, int]:
    """Sanity check: 40 per category, 240 total."""
    counts: dict[str, int] = {}
    for q in ALL_QUERIES:
        counts[q["category"]] = counts.get(q["category"], 0) + 1
    return counts


if __name__ == "__main__":
    counts = category_counts()
    total = sum(counts.values())
    print(f"AE-domain bench corpus: {total} queries across {len(counts)} categories")
    for cat, n in sorted(counts.items()):
        threshold = CATEGORY_THRESHOLDS.get(cat, "?")
        print(f"  {cat:30s} {n:4d}  R@5 threshold: {threshold}")
