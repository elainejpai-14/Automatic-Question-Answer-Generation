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

def make_fill_blank(question, answer, nlp=None):
    if nlp:
        doc = nlp(question)
        # pick first noun or proper noun in question
        key_phrase = next((t.text for t in doc if t.pos_ in ("PROPN", "NOUN")), answer)
    else:
        key_phrase = answer
    if key_phrase.lower() in question.lower():
        return question.replace(key_phrase, "____")
    return question.replace(answer, "____") if answer in question else f"____ ({question})"

def extract_answers(context, nlp):
    """Extract candidate answers from context paragraph."""
    doc = nlp(context)
    answers = list({ent.text for ent in doc.ents if len(ent.text) > 1})
    if not answers:
        # fallback: proper nouns and nouns
        answers = list({t.text for t in doc if t.pos_ in ("PROPN", "NOUN") and len(t.text) > 2})
    if not answers:
        # last fallback: first meaningful sentence
        answers = [context.strip().split(".")[0][:50]]
    return answers

STATIC_DISTRACTORS = {
    "PERSON": ["John Doe", "Jane Smith", "Albert Brown"],
    "ORG": ["Company A", "Organization B", "Institute C"],
    "GPE": ["London", "Paris", "Tokyo"],
    "DATE": ["1990", "2000", "2010"],
    "DEFAULT": ["Option_0", "Option_1", "Option_2"]
}

def make_mcq(question, correct_answer, context, num_options=4, nlp=None):
    candidates = []
    if nlp:
        doc = nlp(context)
        correct_label = None
        for ent in doc.ents:
            if ent.text.lower() == correct_answer.lower():
                correct_label = ent.label_
                break
        candidates = [ent.text for ent in doc.ents 
                      if ent.text.lower() != correct_answer.lower() 
                      and (correct_label is None or ent.label_ == correct_label)]
    random.shuffle(candidates)
    distractors = candidates[:max(0, num_options - 1)]
    if len(distractors) < num_options - 1:
        pool = STATIC_DISTRACTORS.get(correct_label, STATIC_DISTRACTORS["DEFAULT"])
        distractors += pool[:max(0, num_options - 1 - len(distractors))]
    options = distractors + [correct_answer]
    random.shuffle(options)
    return question, options

def make_true_false(question, answer, context, nlp=None):
    is_true = random.random() > 0.5
    used_answer = answer
    if not is_true and nlp:
        doc = nlp(context)
        # choose entity of same type
        answer_label = None
        for ent in doc.ents:
            if ent.text.lower() == answer.lower():
                answer_label = ent.label_
                break
        wrongs = [ent.text for ent in doc.ents if ent.text.lower() != answer.lower() 
                  and (answer_label is None or ent.label_ == answer_label)]
        if wrongs:
            used_answer = random.choice(wrongs)
    statement = f"'{used_answer}' is the correct answer to: \"{question}\""
    return statement, is_true

def make_matching_questions(context, nlp=None, num_pairs=4):
    """Generate matching question pairs from paragraph entities."""
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
    # Repeat answers if fewer than max_qs
    idx = 0
    while len(inputs) < max_qs:
        ans = answers[idx % len(answers)]
        prepared = prepare_qg_input(context, ans)
        if prepared:
            inputs.append(prepared)
            keep_pairs.append(ans)
        idx += 1

    inputs = inputs[:max_qs]
    keep_pairs = keep_pairs[:max_qs]

    # Generate questions
    enc = tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
    outs = qg_model.generate(**enc, max_length=64, num_beams=4)
    questions = [tokenizer.decode(o, skip_special_tokens=True) for o in outs]

    # Verify answers
    batch = [{"question": q, "context": context} for q in questions]
    res = qa_pipe(batch)
    if isinstance(res, dict):
        res = [res]
    verified_answers = [r.get("answer", "") for r in res]

    return list(zip(questions, keep_pairs, verified_answers))

def generate_all_types(context, max_qs=DEFAULT_MAX_Q):
    """Generate WH, Fill-Blank, MCQ, True/False, Matching questions."""
    load_models()
    nlp = MODEL_STATE["nlp"]

    # Step 1: Extract answers
    answers = extract_answers(context, nlp)

    # Step 2: WH questions + verified answers
    wh_pairs = generate_wh_questions(context, answers, max_qs=max_qs)

    # Step 3: Prepare all question types
    rows = []
    for (q, gold_a, pred_a) in wh_pairs:
        # WH
        rows.append({"Type": "WH", "Question": q, "Answer / Gold": pred_a or gold_a, "Options": ""})
        # Fill-Blank
        fb_q = make_fill_blank(q, gold_a, nlp=nlp)
        rows.append({"Type": "Fill-Blank", "Question": fb_q, "Answer / Gold": gold_a, "Options": ""})
        # MCQ
        mcq_q, mcq_opts = make_mcq(q, gold_a, context, nlp=nlp)
        rows.append({"Type": "MCQ", "Question": mcq_q, "Answer / Gold": gold_a, "Options": " | ".join(mcq_opts)})
        # True/False
        tf_stmt, tf_val = make_true_false(q, gold_a, context, nlp=nlp)
        rows.append({"Type": "True/False", "Question": tf_stmt, "Answer / Gold": str(tf_val), "Options": ""})

    # Step 4: Matching questions (once per paragraph)
    matching_pairs = make_matching_questions(context, nlp=nlp)
    if matching_pairs:
        rows.append({
            "Type": "Matching",
            "Question": "",
            "Answer / Gold": "",
            "Options": "; ".join([f"{a} -> {b}" for a, b in matching_pairs])
        })

    # Step 5: Return as DataFrame
    out_df = pd.DataFrame(rows)
    return out_df

# ---------- Streamlit UI ----------
st.title("Automatic Q&A generator")
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
