"""
Twenty-two tickets unlike anything in the training years, written for
Chapter 20 and labelled with the rubric: new wording in English, new
sentences in the three languages Meridian already sees, and languages
it has never received. None comes from the warehouse. They test what
each approach does with input it was not trained on.
"""
from __future__ import annotations

import pandas as pd

PROBES = [
    # kind, body, category, priority
    ("new wording", "the courier left our pallet out in the rain, "
     "every box is wet through", "Delivery", "Normal"),
    ("new wording", "bleach MRD-SAN-041 gives off a strong smell, "
     "three cleaners have headaches", "Quality", "Urgent"),
    ("new wording", "our kitchen cannot open for lunch, we have no "
     "hand soap left at all", "Stock", "Urgent"),
    ("new wording", "the mop heads we got this month fall to bits "
     "the first time they are used", "Quality", "High"),
    ("new wording", "we have been charged two times for the same "
     "delivery", "Billing", "Normal"),
    ("new wording", "can the paper towels go back? we picked the "
     "wrong ones", "Returns", "Normal"),
    ("new wording", "nobody came to pick up the return you arranged "
     "last week", "Returns", "Normal"),
    ("new wording", "we will be switching suppliers when this "
     "agreement runs out", "Account", "Normal"),
    ("known language", "Nos gants sont arrivés déchirés, le carton "
     "était écrasé.", "Delivery", "Normal"),
    ("known language", "Il nous faut du savon aujourd'hui, sinon la "
     "cuisine reste fermée.", "Stock", "Urgent"),
    ("known language", "Merci de mettre à jour notre adresse de "
     "facturation.", "Account", "Low"),
    ("known language", "La factura INV-44821 tiene un cargo "
     "duplicado.", "Billing", "Normal"),
    ("known language", "Faltan tres cajas en el pedido 1182204.",
     "Delivery", "High"),
    ("known language", "Die Lieferung kam völlig durchnässt an, alles "
     "unbrauchbar.", "Delivery", "Normal"),
    ("known language", "Wir möchten MRD-PAC-021 zurückschicken, "
     "falsche Größe bestellt.", "Returns", "Normal"),
    ("new language", "L'ordine 1203341 non è ancora arrivato.",
     "Delivery", "Normal"),
    ("new language", "Il detergente MRD-SAN-044 ha fatto stare male "
     "due persone.", "Quality", "Urgent"),
    ("new language", "A fatura INV-55120 foi cobrada duas vezes.",
     "Billing", "Normal"),
    ("new language", "O produto MRD-CLE-006 está com defeito, o lote "
     "inteiro.", "Quality", "High"),
    ("new language", "Bestelling 1302217 is nog steeds niet "
     "aangekomen.", "Delivery", "Normal"),
    ("new language", "Kunnen jullie ons adres aanpassen?", "Account",
     "Low"),
    ("new language", "Zamówienie 998120 jeszcze nie dotarło.",
     "Delivery", "Normal"),
]


def probes() -> pd.DataFrame:
    return pd.DataFrame(PROBES, columns=["kind", "body", "category",
                                         "priority"])
