# app.py
import os
import random
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import spacy
from rapidfuzz import fuzz
import streamlit as st

# ---------- Config ----------
DEFAULT_MAX_Q = 3
MAX_MCQ_OPTIONS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Helper functions ----------
def highlight_answer(context, answer):
    start_idx = context.lower().find(answer.lower())
    if start_idx == -1:
        return None
    end_idx = start_idx + len(answer)
    return context[:start_idx] + "<hl> " + answer + " <hl>" + context[end_idx:]

def prepare_qg_input(context, answer):
    highlighted = highlight_answer(context, answer)
    if highlighted:
        return f"generate question: {highlighted}"
    return None

def compute_metrics(pred, gold):
    em = int(pred.strip().lower() == gold.strip().lower())
    f1 = fuzz.token_sort_ratio(pred, gold) / 100
    return em, f1

def make_fill_blank(question, answer):
    if answer.lower() in question.lower():
        return question.replace(answer, "____")
    return f"____ ({question})"

def make_mcq(question, correct_answer, context, num_options=MAX_MCQ_OPTIONS, nlp=None):
    candidates = []
    if nlp:
        doc = nlp(context)
        candidates = list({ent.text for ent in doc.ents if ent.text.lower() != correct_answer.lower()})
    random.shuffle(candidates)
    distractors = candidates[: max(0, num_options - 1)]
    if len(distractors) < num_options - 1:
        distractors += [f"Option_{i}" for i in range(len(distractors), num_options - 1)]
    options = distractors + [correct_answer]
    random.shuffle(options)
    return question, options

def make_true_false(question, answer, context, nlp=None):
    is_true = random.random() > 0.5
    used_answer = answer
    if not is_true and nlp:
        doc = nlp(context)
        wrongs = [ent.text for ent in doc.ents if ent.text.lower() != answer.lower()]
        if wrongs:
            used_answer = random.choice(wrongs)
    statement = f"'{used_answer}' is the correct answer to: \"{question}\""
    return statement, is_true

def make_matching(context, num_pairs=4, nlp=None):
    if not nlp:
        return []
    doc = nlp(context)
    entities = list({(ent.text, ent.label_) for ent in doc.ents})
    random.shuffle(entities)
    return entities[:num_pairs]

# ---------- Model loading ----------
MODEL_STATE = {"qg_tokenizer": None, "qg_model": None, "qa_pipeline": None, "nlp": None}

def load_models():
    if MODEL_STATE["nlp"] is None:
        try:
            MODEL_STATE["nlp"] = spacy.load("en_core_web_sm")
        except Exception:
            MODEL_STATE["nlp"] = spacy.blank("en")

    if MODEL_STATE["qg_tokenizer"] is None:
        MODEL_STATE["qg_tokenizer"] = T5Tokenizer.from_pretrained("valhalla/t5-base-qg-hl")
    if MODEL_STATE["qg_model"] is None:
        MODEL_STATE["qg_model"] = T5ForConditionalGeneration.from_pretrained("valhalla/t5-base-qg-hl").to(DEVICE)
    if MODEL_STATE["qa_pipeline"] is None:
        device_id = 0 if DEVICE == "cuda" else -1
        MODEL_STATE["qa_pipeline"] = pipeline("question-answering", model="deepset/roberta-base-squad2", device=device_id)

def generate_wh_questions(context, answers, max_qs=DEFAULT_MAX_Q):
    load_models()
    tokenizer = MODEL_STATE["qg_tokenizer"]
    qg_model = MODEL_STATE["qg_model"]
    qa_pipe = MODEL_STATE["qa_pipeline"]

    inputs, keep_pairs = [], []
    for ans in answers:
        prepared = prepare_qg_input(context, ans)
        if prepared:
            inputs.append(prepared)
            keep_pairs.append(ans)

    inputs = inputs[:max_qs]
    keep_pairs = keep_pairs[:max_qs]

    questions, verified_answers = [], []
    if inputs:
        enc = tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        outs = qg_model.generate(**enc, max_length=64, num_beams=4)
        questions = [tokenizer.decode(o, skip_special_tokens=True) for o in outs]

        batch = [{"question": q, "context": context} for q in questions]
        res = qa_pipe(batch)
        if isinstance(res, dict):
            res = [res]
        verified_answers = [r.get("answer", "") for r in res]

    return list(zip(questions, keep_pairs, verified_answers))

def generate_all_types(context, max_qs=DEFAULT_MAX_Q):
    load_models()
    nlp = MODEL_STATE["nlp"]

    doc = nlp(context)
    answers = [ent.text for ent in doc.ents]
    if not answers:
        tokens = [t.text for t in doc if t.pos_ in ("PROPN", "NOUN") and len(t.text) > 2]
        answers = list(dict.fromkeys(tokens))
    if not answers:
        answers = [context.strip().split(".")[0][:50]]

    wh_pairs = generate_wh_questions(context, answers, max_qs=max_qs)

    rows = []
    for (q, gold_a, pred_a) in wh_pairs:
        rows.append({"Type": "WH", "Question": q, "Gold Answer": gold_a, "Verified Answer": pred_a})
        rows.append({"Type": "Fill-Blank", "Question": make_fill_blank(q, gold_a), "Answer": gold_a})
        mcq_q, mcq_opts = make_mcq(q, gold_a, context, nlp=nlp)
        rows.append({"Type": "MCQ", "Question": mcq_q, "Answer": gold_a, "Options": mcq_opts})
        tf_stmt, tf_val = make_true_false(q, gold_a, context, nlp=nlp)
        rows.append({"Type": "True/False", "Question": tf_stmt, "Answer": str(tf_val)})
        pairs = make_matching(context, nlp=nlp)
        rows.append({"Type": "Matching", "Pairs": pairs})

    df_rows = []
    for r in rows:
        if r["Type"] == "MCQ":
            df_rows.append({
                "Type": r["Type"],
                "Question": r["Question"],
                "Answer / Gold": r.get("Answer", ""),
                "Options": " | ".join(r.get("Options", []))
            })
        elif r["Type"] == "Matching":
            df_rows.append({
                "Type": r["Type"],
                "Question": "",
                "Answer / Gold": "",
                "Options": "; ".join([f"{a} -> {b}" for a,b in r.get("Pairs", [])])
            })
        else:
            df_rows.append({
                "Type": r["Type"],
                "Question": r.get("Question", ""),
                "Answer / Gold": r.get("Answer", r.get("Gold Answer", r.get("Verified Answer", ""))),
                "Options": ""
            })

    out_df = pd.DataFrame(df_rows)
    return out_df

# ---------- Streamlit UI ----------
st.title("QG & QA — multi-type generator")
st.markdown("Generate WH, Fill-Blank, MCQ, True/False, and Matching questions from a paragraph.")

context_in = st.text_area("Context paragraph", height=150, placeholder="Paste a paragraph here...")
max_qs = st.slider("Max WH questions to generate", 1, 6, DEFAULT_MAX_Q)

if st.button("Generate"):
    if not context_in.strip():
        st.warning("Please enter some context text.")
    else:
        with st.spinner("Generating questions..."):
            out_df = generate_all_types(context_in, max_qs)
            st.dataframe(out_df)
            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv_bytes, file_name="generated_qa.csv", mime="text/csv")
