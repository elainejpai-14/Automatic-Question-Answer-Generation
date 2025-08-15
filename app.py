# app.py
# Streamlit offline question generator (improved)

import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import random
import nltk
import os

# Ensure a local nltk_data directory exists
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download the standard punkt and stopwords
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_path, quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir=nltk_data_path, quiet=True)

# Patch: punkt_tab does not exist; redirect to punkt
from nltk.tokenize import sent_tokenize
nltk.data.path.append(nltk_data_path)

# --- Load T5 model for WH question generation ---
@st.cache_resource(show_spinner=True)
def load_model():
    model_name = "valhalla/t5-small-qg-hl"  # small and fast for offline use
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# --- Helper functions ---
def generate_wh_question(sentence):
    input_text = "generate question: " + sentence
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=64, num_beams=4, early_stopping=True)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

def generate_true_false(sentence):
    negation_patterns = [
        (" is ", " is not "), (" are ", " are not "),
        (" has ", " has not "), (" have ", " have not "),
        (" was ", " was not "), (" were ", " were not ")
    ]
    if random.random() > 0.5:
        tf_answer = "False"
        for old, new in negation_patterns:
            if old in sentence:
                sentence = sentence.replace(old, new)
                break
    else:
        tf_answer = "True"
    return sentence, tf_answer

def generate_fill_blank(sentence):
    words = sentence.split()
    candidate_words = [w for w in words if w.lower() not in stop_words and len(w) > 2]
    if not candidate_words:
        return None, None
    answer = random.choice(candidate_words)
    question = sentence.replace(answer, "_____", 1)
    return question, answer

def generate_mcq(sentence):
    words = list(set(sentence.split()))
    words = [w for w in words if w.lower() not in stop_words and len(w) > 2]
    if not words:
        return None, None, None
    answer = random.choice(words)
    words.remove(answer)
    options = [answer]
    dummy_options = ["OptionA", "OptionB", "OptionC", "OptionD"]
    fake_options = random.sample(words, k=min(3, len(words))) if len(words) >= 3 else dummy_options[:3]
    options.extend(fake_options)
    random.shuffle(options)
    question = sentence.replace(answer, "_____", 1)
    return question, options, answer

def generate_matching(sentences):
    left, right = [], []
    for sent in sentences:
        words = nltk.pos_tag(sent.split())
        nouns = [w for w, pos in words if pos.startswith("NN")]
        if nouns:
            left.append(sent)
            right.append(random.choice(nouns))
    if not left:
        return []
    shuffled_right = right.copy()
    random.shuffle(shuffled_right)
    pairs = [{"left": l, "right": r} for l, r in zip(left, shuffled_right)]
    return pairs

def generate_questions_from_paragraph(paragraph, max_wh=5):
    sentences = sent_tokenize(paragraph)
    questions = {"WH": [], "TrueFalse": [], "FillBlank": [], "MCQ": [], "Matching": []}

    for idx, sent in enumerate(sentences):
        if idx < max_wh:
            questions["WH"].append({"question": generate_wh_question(sent), "answer": sent})
        tf_question, tf_answer = generate_true_false(sent)
        questions["TrueFalse"].append({"question": tf_question, "answer": tf_answer})
        fb_question, fb_answer = generate_fill_blank(sent)
        if fb_question:
            questions["FillBlank"].append({"question": fb_question, "answer": fb_answer})
        mcq_question, options, mcq_answer = generate_mcq(sent)
        if mcq_question:
            questions["MCQ"].append({"question": mcq_question, "options": options, "answer": mcq_answer})

    questions["Matching"] = generate_matching(sentences)
    return questions

# --- Streamlit UI ---
st.set_page_config(page_title="Offline Question Generator", layout="wide")
st.title("Automatic Q&A Generator")
st.markdown("Enter a paragraph below, and the app will generate multiple types of questions.")

paragraph_input = st.text_area("Paste your paragraph here:", height=200)

if st.button("Generate Questions"):
    if paragraph_input.strip() == "":
        st.warning("Please enter a paragraph!")
    else:
        with st.spinner("Generating questions..."):
            qa_pairs = generate_questions_from_paragraph(paragraph_input)

        for qtype, qlist in qa_pairs.items():
            with st.expander(f"{qtype} Questions ({len(qlist)})", expanded=False):
                for idx, q in enumerate(qlist, 1):
                    if qtype == "MCQ":
                        st.markdown(f"**Q{idx}:** {q['question']}")
                        for opt in q['options']:
                            st.markdown(f"- {opt}")
                        st.markdown(f"**Answer:** {q['answer']}")
                    elif qtype == "Matching":
                        st.markdown(f"**Pair {idx}:** Left -> {q['left']} | Right -> {q['right']}")
                    else:
                        st.markdown(f"**Q{idx}:** {q['question']}")
                        st.markdown(f"**Answer:** {q['answer']}")
