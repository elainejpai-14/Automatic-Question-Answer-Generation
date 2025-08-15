# app.py
# Streamlit offline question generator

import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import random
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

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
    if random.random() > 0.5:
        tf_answer = "False"
        if " is " in sentence:
            sentence = sentence.replace(" is ", " is not ")
        elif " are " in sentence:
            sentence = sentence.replace(" are ", " are not ")
    else:
        tf_answer = "True"
    return sentence, tf_answer

def generate_fill_blank(sentence):
    words = sentence.split()
    if len(words) < 4:
        return sentence, ""
    blank_idx = random.randint(0, len(words)-1)
    answer = words[blank_idx]
    words[blank_idx] = "_____"
    question = " ".join(words)
    return question, answer

def generate_mcq(sentence):
    words = sentence.split()
    if len(words) < 4:
        return sentence, ["A", "B", "C", "D"], "A"
    answer = random.choice(words)
    options = [answer]
    while len(options) < 4:
        fake_option = random.choice(words)
        if fake_option not in options:
            options.append(fake_option)
    random.shuffle(options)
    question = sentence.replace(answer, "_____", 1)
    return question, options, answer

def generate_matching(sentences):
    """
    Generate simple matching pairs: key -> value randomly
    For demo, left = sentence, right = first noun chunk / word
    """
    left = []
    right = []
    for sent in sentences:
        words = sent.split()
        if words:
            left.append(sent)
            right.append(random.choice(words))
    shuffled_right = right.copy()
    random.shuffle(shuffled_right)
    pairs = [{"left": l, "right": r} for l, r in zip(left, shuffled_right)]
    return pairs

def generate_questions_from_paragraph(paragraph):
    sentences = sent_tokenize(paragraph)
    questions = {"WH": [], "TrueFalse": [], "FillBlank": [], "MCQ": [], "Matching": []}

    for sent in sentences:
        questions["WH"].append({"question": generate_wh_question(sent), "answer": sent})
        tf_question, tf_answer = generate_true_false(sent)
        questions["TrueFalse"].append({"question": tf_question, "answer": tf_answer})
        fb_question, fb_answer = generate_fill_blank(sent)
        questions["FillBlank"].append({"question": fb_question, "answer": fb_answer})
        mcq_question, options, mcq_answer = generate_mcq(sent)
        questions["MCQ"].append({"question": mcq_question, "options": options, "answer": mcq_answer})
    
    questions["Matching"] = generate_matching(sentences)
    return questions

# --- Streamlit UI ---
st.set_page_config(page_title="Offline Question Generator", layout="wide")
st.title("📚 Offline Question Generator")

st.markdown("Enter a paragraph below, and the app will generate multiple types of questions.")

paragraph_input = st.text_area("Paste your paragraph here:", height=200)

if st.button("Generate Questions"):
    if paragraph_input.strip() == "":
        st.warning("Please enter a paragraph!")
    else:
        with st.spinner("Generating questions..."):
            qa_pairs = generate_questions_from_paragraph(paragraph_input)
        
        # Display results
        for qtype, qlist in qa_pairs.items():
            st.subheader(f"{qtype} Questions")
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
