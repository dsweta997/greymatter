"""
Build a stratified ~100-question retrieval eval set, GROUNDED against the real corpus.

Philosophy (per proposal Phase 1.4): the eval set is the measuring stick — it must be
trustworthy. So questions are hand-authored here, but every `relevant_doc_ids` and
`must_contain` is verified against corpus/raw: a question only gets real doc ids that
actually contain its anchor terms. Anything that can't be grounded is written with
`_needs_review: true` and empty ids, so nothing is silently wrong.

Output: corpus/eval/eval_set.jsonl  (one JSON object per line)

Run:  python data_pipeline/02_build_eval_set.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> import shared pkgs

from common import load_config  # noqa: E402

_CFG = load_config()
RAW = Path(_CFG["paths"]["corpus_raw"])
OUT = Path(_CFG["paths"]["corpus_eval"]) / "eval_set.jsonl"

# --------------------------------------------------------------------------- #
# Question bank — stratified by subject (roughly mapped to source) and type.
# type: student | exam | process     Each has anchor terms used for grounding.
# Anchors should be distinctive phrases the answering doc will contain.
# --------------------------------------------------------------------------- #
QUESTIONS = [
    # ---------- Anatomy & Physiology (~25) ----------
    (
        "Why does the left ventricle have a thicker wall than the right ventricle?",
        ["left ventricle", "thicker", "systemic"],
        "student",
        "anatomy_physiology",
    ),
    (
        "What is the function of the sinoatrial node in the heart?",
        ["sinoatrial node", "pacemaker"],
        "student",
        "anatomy_physiology",
    ),
    (
        "How does the sliding filament model of muscle contraction work?",
        ["sliding filament", "actin", "myosin"],
        "student",
        "anatomy_physiology",
    ),
    (
        "What is the role of the loop of Henle in the nephron?",
        ["loop of Henle", "concentration"],
        "exam",
        "anatomy_physiology",
    ),
    (
        "Describe the pathway of blood through the pulmonary circuit.",
        ["pulmonary", "right ventricle", "lungs"],
        "process",
        "anatomy_physiology",
    ),
    (
        "What are the three types of fibrous joints?",
        ["suture", "syndesmosis", "gomphosis"],
        "exam",
        "anatomy_physiology",
    ),
    (
        "How is oxygen transported in the blood?",
        ["hemoglobin", "oxygen"],
        "student",
        "anatomy_physiology",
    ),
    (
        "What is the difference between afferent and efferent neurons?",
        ["sensory neuron", "motor neuron"],
        "student",
        "anatomy_physiology",
    ),
    (
        "What happens during depolarization of a neuron?",
        ["depolarization", "sodium"],
        "exam",
        "anatomy_physiology",
    ),
    (
        "What is the function of the alveoli in gas exchange?",
        ["alveoli", "gas exchange"],
        "student",
        "anatomy_physiology",
    ),
    (
        "How does the countercurrent mechanism concentrate urine?",
        ["countercurrent", "urine"],
        "exam",
        "anatomy_physiology",
    ),
    (
        "What is the role of insulin in blood glucose regulation?",
        ["insulin", "glucose"],
        "student",
        "anatomy_physiology",
    ),
    (
        "Trace the flow of a nerve impulse across a synapse.",
        ["synapse", "neurotransmitter"],
        "process",
        "anatomy_physiology",
    ),
    (
        "What are the phases of the cardiac cycle?",
        ["cardiac cycle", "systole", "diastole"],
        "exam",
        "anatomy_physiology",
    ),
    (
        "How do skeletal muscles produce movement at joints?",
        ["skeletal muscle", "contraction"],
        "student",
        "anatomy_physiology",
    ),
    (
        "What is the function of the myelin sheath?",
        ["myelin", "conduction"],
        "student",
        "anatomy_physiology",
    ),
    (
        "Describe how the kidney filters blood to form urine.",
        ["glomerulus", "filtration"],
        "process",
        "anatomy_physiology",
    ),
    (
        "What is homeostasis and how is it maintained?",
        ["homeostasis", "feedback"],
        "student",
        "anatomy_physiology",
    ),
    (
        "What are the functions of the different regions of the nephron?",
        ["nephron", "reabsorption"],
        "exam",
        "anatomy_physiology",
    ),
    (
        "How does blood pressure get regulated by the body?",
        ["blood pressure", "baroreceptor"],
        "exam",
        "anatomy_physiology",
    ),
    (
        "What is the structure and function of a sarcomere?",
        ["sarcomere", "muscle"],
        "exam",
        "anatomy_physiology",
    ),
    (
        "How do hormones of the endocrine system reach their targets?",
        ["hormone", "bloodstream"],
        "student",
        "anatomy_physiology",
    ),
    (
        "What role does the diaphragm play in breathing?",
        ["diaphragm", "breathing"],
        "student",
        "anatomy_physiology",
    ),
    (
        "What is the difference between arteries and veins?",
        ["arteries", "veins"],
        "student",
        "anatomy_physiology",
    ),
    (
        "How does the resting membrane potential arise in a cell?",
        ["resting membrane potential", "potassium"],
        "exam",
        "anatomy_physiology",
    ),
    # ---------- Microbiology (~15) ----------
    (
        "What is the difference between Gram-positive and Gram-negative bacteria?",
        ["Gram-positive", "Gram-negative", "peptidoglycan"],
        "exam",
        "microbiology",
    ),
    (
        "How do antibiotics inhibit bacterial cell wall synthesis?",
        ["cell wall", "antibiotic"],
        "exam",
        "microbiology",
    ),
    (
        "What are the stages of the bacterial growth curve?",
        ["growth curve", "lag", "exponential"],
        "exam",
        "microbiology",
    ),
    (
        "How does the process of binary fission work in bacteria?",
        ["binary fission"],
        "process",
        "microbiology",
    ),
    (
        "What is the structure of a bacteriophage?",
        ["bacteriophage", "capsid"],
        "student",
        "microbiology",
    ),
    (
        "How do viruses replicate inside host cells?",
        ["virus", "replication", "host"],
        "process",
        "microbiology",
    ),
    (
        "What is the role of endospores in bacterial survival?",
        ["endospore"],
        "student",
        "microbiology",
    ),
    (
        "How does the Gram stain procedure work?",
        ["Gram stain", "crystal violet"],
        "process",
        "microbiology",
    ),
    (
        "What are the main differences between bacteria and archaea?",
        ["archaea", "bacteria"],
        "exam",
        "microbiology",
    ),
    (
        "How do antimicrobial drugs achieve selective toxicity?",
        ["selective toxicity", "antimicrobial"],
        "exam",
        "microbiology",
    ),
    (
        "What is horizontal gene transfer in bacteria?",
        ["horizontal gene transfer", "conjugation"],
        "exam",
        "microbiology",
    ),
    (
        "How does the immune system respond to a bacterial infection?",
        ["immune", "infection"],
        "student",
        "microbiology",
    ),
    (
        "What is the difference between exotoxins and endotoxins?",
        ["exotoxin", "endotoxin"],
        "exam",
        "microbiology",
    ),
    ("How do vaccines produce immunity?", ["vaccine", "immunity"], "student", "microbiology"),
    (
        "What are the mechanisms of antibiotic resistance?",
        ["antibiotic resistance"],
        "exam",
        "microbiology",
    ),
    # ---------- Cell & Molecular Biology (~15) ----------
    (
        "How does DNA replication occur?",
        ["DNA replication", "polymerase"],
        "process",
        "cell_biology",
    ),
    (
        "What is the process of protein synthesis from DNA?",
        ["transcription", "translation"],
        "process",
        "cell_biology",
    ),
    (
        "What are the phases of mitosis?",
        ["mitosis", "prophase", "metaphase"],
        "exam",
        "cell_biology",
    ),
    (
        "How does the cell membrane control what enters and exits the cell?",
        ["cell membrane", "transport"],
        "student",
        "cell_biology",
    ),
    (
        "What is the function of mitochondria in the cell?",
        ["mitochondria", "ATP"],
        "student",
        "cell_biology",
    ),
    (
        "How does cellular respiration produce ATP?",
        ["cellular respiration", "ATP"],
        "process",
        "cell_biology",
    ),
    (
        "What is the role of enzymes in biochemical reactions?",
        ["enzyme", "substrate"],
        "student",
        "cell_biology",
    ),
    (
        "How do cells differentiate into specialized types?",
        ["differentiation", "cell"],
        "exam",
        "cell_biology",
    ),
    (
        "What happens during meiosis and how does it differ from mitosis?",
        ["meiosis", "mitosis"],
        "exam",
        "cell_biology",
    ),
    (
        "What is the structure of the DNA double helix?",
        ["double helix", "nucleotide"],
        "student",
        "cell_biology",
    ),
    ("How does DNA repair fix damage to the genome?", ["DNA repair"], "exam", "cell_biology"),
    (
        "What is the role of ribosomes in the cell?",
        ["ribosome", "protein"],
        "student",
        "cell_biology",
    ),
    ("How do cells regulate the cell cycle?", ["cell cycle", "checkpoint"], "exam", "cell_biology"),
    (
        "What is photosynthesis and where does it occur?",
        ["photosynthesis", "chloroplast"],
        "student",
        "cell_biology",
    ),
    (
        "How does osmosis move water across a membrane?",
        ["osmosis", "water"],
        "student",
        "cell_biology",
    ),
    # ---------- Nursing fundamentals (~15) ----------
    ("What are the vital signs a nurse routinely measures?", ["vital signs"], "student", "nursing"),
    (
        "What is the nursing process and its steps?",
        ["nursing process", "assessment"],
        "exam",
        "nursing",
    ),
    (
        "How should a nurse approach patient assessment?",
        ["patient assessment"],
        "student",
        "nursing",
    ),
    (
        "What are the principles of infection control in nursing?",
        ["infection control"],
        "exam",
        "nursing",
    ),
    ("How does a nurse care for patients with a disability?", ["disability"], "student", "nursing"),
    (
        "What is the importance of hand hygiene in healthcare?",
        ["hand hygiene"],
        "student",
        "nursing",
    ),
    ("How is pain assessed and managed in patients?", ["pain", "assessment"], "student", "nursing"),
    ("What are the stages of wound healing?", ["wound healing"], "exam", "nursing"),
    ("How should a nurse document patient care?", ["documentation"], "student", "nursing"),
    (
        "What are the principles of patient-centered care?",
        ["patient-centered"],
        "student",
        "nursing",
    ),
    ("How does a nurse maintain patient safety?", ["patient safety"], "student", "nursing"),
    (
        "What is the role of therapeutic communication in nursing?",
        ["therapeutic communication"],
        "exam",
        "nursing",
    ),
    (
        "How are medications administered safely by nurses?",
        ["medication administration"],
        "exam",
        "nursing",
    ),
    ("What cultural considerations affect patient care?", ["cultural"], "student", "nursing"),
    (
        "How does a nurse assess a patient's nutritional status?",
        ["nutrition", "assessment"],
        "student",
        "nursing",
    ),
    # ---------- General medical Q&A (MedQuAD register) (~20) ----------
    ("What is diabetes?", ["diabetes"], "student", "general_medical"),
    (
        "What causes high blood pressure?",
        ["blood pressure", "hypertension"],
        "student",
        "general_medical",
    ),
    ("What are the symptoms of asthma?", ["asthma"], "student", "general_medical"),
    ("What is anemia and what causes it?", ["anemia"], "student", "general_medical"),
    ("What is the outlook for people with dyslexia?", ["dyslexia"], "student", "general_medical"),
    ("What is glaucoma?", ["glaucoma"], "student", "general_medical"),
    ("What causes osteoporosis?", ["osteoporosis"], "student", "general_medical"),
    (
        "What is a stroke and how does it affect the brain?",
        ["stroke"],
        "student",
        "general_medical",
    ),
    ("What is chronic kidney disease?", ["kidney disease"], "student", "general_medical"),
    ("What are the functions of the thyroid gland?", ["thyroid"], "student", "general_medical"),
    ("What is epilepsy?", ["epilepsy"], "student", "general_medical"),
    ("What causes migraines?", ["migraine"], "student", "general_medical"),
    ("What is arthritis and what are its types?", ["arthritis"], "student", "general_medical"),
    (
        "What is the difference between type 1 and type 2 diabetes?",
        ["type 1", "type 2", "diabetes"],
        "exam",
        "general_medical",
    ),
    ("What is hepatitis?", ["hepatitis"], "student", "general_medical"),
    ("What causes anemia in chronic disease?", ["anemia"], "student", "general_medical"),
    ("What is Parkinson's disease?", ["Parkinson"], "student", "general_medical"),
    (
        "What is the role of the immune system in allergies?",
        ["allergy", "immune"],
        "student",
        "general_medical",
    ),
    ("What is cystic fibrosis?", ["cystic fibrosis"], "student", "general_medical"),
    (
        "What causes a urinary tract infection?",
        ["urinary tract infection"],
        "student",
        "general_medical",
    ),
    # ---------- Cross-source / process (multi-chunk) (~10) ----------
    (
        "Trace a drop of blood from the right atrium to the aorta.",
        ["right atrium", "aorta"],
        "process",
        "cross_source",
    ),
    (
        "Explain how a bacterial infection triggers an immune response, from entry to clearance.",
        ["infection", "immune response"],
        "process",
        "cross_source",
    ),
    (
        "Describe the journey of oxygen from the air to a body cell.",
        ["oxygen", "alveoli", "cell"],
        "process",
        "cross_source",
    ),
    (
        "How does the body regulate blood sugar after a meal?",
        ["glucose", "insulin"],
        "process",
        "cross_source",
    ),
    (
        "Explain how a nerve signal leads to muscle contraction.",
        ["nerve", "muscle contraction"],
        "process",
        "cross_source",
    ),
    (
        "Trace the path of food through the digestive system.",
        ["digestive", "stomach", "intestine"],
        "process",
        "cross_source",
    ),
    (
        "Describe how the kidneys help maintain blood pH.",
        ["kidney", "pH"],
        "process",
        "cross_source",
    ),
    (
        "Explain the steps of the inflammatory response to tissue injury.",
        ["inflammation", "injury"],
        "process",
        "cross_source",
    ),
    (
        "How does a genetic mutation lead to a change in a protein?",
        ["mutation", "protein"],
        "process",
        "cross_source",
    ),
    (
        "Describe how a vaccine prepares the immune system for future infection.",
        ["vaccine", "immune"],
        "process",
        "cross_source",
    ),
]


# --------------------------------------------------------------------------- #
def load_corpus():
    docs = []
    for f in RAW.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        docs.append((d["id"], d["source"], d["text"], d["text"].lower()))
    return docs


# Scope each question's grounding to the source(s) most likely to answer it, so a
# question can't match a doc from an unrelated subject on a coincidental keyword.
SUBJECT_SOURCES = {
    "anatomy_physiology": {"openstax_ap2e"},
    "microbiology": {"openstax_microbiology"},
    "cell_biology": {"openstax_biology2e"},
    "nursing": {"openstax_fundamentals_nursing"},
    "general_medical": {"medquad"},
    "cross_source": None,  # any source
}


def ground(anchors, subject, docs, max_ids=3):
    """Find real doc ids that are ABOUT the anchor terms, scoped to the subject's source.

    Requires ALL anchors present, then ranks by total anchor frequency (a doc that is
    about the topic mentions it repeatedly). Returns (ids, present, confident).
    """
    anchors_l = [a.lower() for a in anchors]
    allowed = SUBJECT_SOURCES.get(subject)
    scored = []
    for did, source, text, low in docs:
        if allowed is not None and source not in allowed:
            continue
        present_ct = sum(1 for a in anchors_l if a in low)
        if present_ct < len(anchors_l):
            continue  # require ALL anchors -> much stricter than "any"
        freq = sum(low.count(a) for a in anchors_l)  # topicality signal
        scored.append((freq, did, low))
    if not scored:
        return [], anchors[:1], False
    scored.sort(key=lambda s: s[0], reverse=True)
    top = scored[:max_ids]
    ids = [s[1] for s in top]
    present = [a for a in anchors if all(a.lower() in s[2] for s in top)]
    return ids, (present or anchors), True


def main():
    docs = load_corpus()
    print(f"loaded {len(docs)} corpus docs")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    rows, needs_review = [], 0
    by_subject, by_type = Counter(), Counter()
    for q, anchors, qtype, subject in QUESTIONS:
        ids, present, confident = ground(anchors, subject, docs)
        row = {
            "question": q,
            "relevant_doc_ids": ids,
            "must_contain": present,
            "type": qtype,
            "subject": subject,
        }
        # Single-subject questions ground reliably by scoped keyword search.
        # Cross-source/process questions span multiple docs and need semantic
        # retrieval + human judgment (Phase 2/3) — flag their ids as provisional.
        if not confident or not ids or subject == "cross_source":
            row["_needs_review"] = True
            needs_review += 1
        rows.append(row)
        by_subject[subject] += 1
        by_type[qtype] += 1

    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(rows)} questions -> {OUT}")
    print(f"grounded: {len(rows) - needs_review} | needs review (no match): {needs_review}")
    print("\nby subject:")
    for k, v in by_subject.most_common():
        print(f"  {k:20} {v}")
    print("\nby type:")
    for k, v in by_type.most_common():
        print(f"  {k:10} {v}")


if __name__ == "__main__":
    main()
