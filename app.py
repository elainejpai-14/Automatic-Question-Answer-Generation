# app.py
# Streamlit bilingual question generator (English/Kannada) with WH metrics

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import random
import re
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
import sacrebleu
from nltk.tokenize import word_tokenize
import nltk
from nltk import pos_tag

# Ensure NLTK resources are downloaded
def _ensure_nltk_data():
    needed = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab/english", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, pkg in needed:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg)
            
_ensure_nltk_data()

# --- Streamlit page config ---
st.set_page_config(page_title="Automatic Q&A Generator", layout="wide")

# --- UI text dictionary ---
UI_TEXT = {
    "en": {
        "title": "Automatic Q&A Generator",
        "desc": "Enter a paragraph below, and the app will generate multiple types of questions.",
        "input": "Paste your paragraph here:",
        "generate_btn": "Generate Questions",
        "warning": "Please enter a paragraph!",
        "download": "Download as CSV"
    },
    "kn": {
        "title": "ಸ್ವಯಂಚಾಲಿತ ಪ್ರಶ್ನೆ-ಉತ್ತರ ತಯಾರಕ",
        "desc": "ಕೆಳಗಿನ ಪ್ಯಾರಾಗ್ರಾಫ್ ಅನ್ನು ನಮೂದಿಸಿ, ಅಪ್ಲಿಕೇಶನ್ ವಿವಿಧ ರೀತಿಯ ಪ್ರಶ್ನೆಗಳನ್ನು ತಯಾರಿಸುತ್ತದೆ.",
        "input": "ನಿಮ್ಮ ಪ್ಯಾರಾಗ್ರಾಫ್ ಅನ್ನು ಇಲ್ಲಿ ಹಾಕಿ:",
        "generate_btn": "ಪ್ರಶ್ನೆಗಳನ್ನು ತಯಾರಿಸಿ",
        "warning": "ದಯವಿಟ್ಟು ಪ್ಯಾರಾಗ್ರಾಫ್ ನಮೂದಿಸಿ!",
        "download": "CSV ಆಗಿ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ"
    }
}

# --- Stopwords ---
STOP_WORDS_EN = set([
    "a", "an", "the", "and", "or", "but", "if", "while", "with",
    "is", "are", "was", "were", "has", "have", "had", "of", "in", "on", "for",
    "to", "from", "by", "as", "at", "that", "this", "it"
])

STOP_WORDS_KN = set([
    "ಆಗ", "ಅವರು", "ಅನ್ನು", "ಅಲ್ಲಿ", "ಇದ್ದ", "ಇದೆ", "ಮತ್ತು", "ಅದನ್ನು",
    "ಅಥವಾ", "ಆದರೆ", "ಇದೆ", "ಇದ್ದವು", "ಇದನ್ನು", "ಅವರು"
])

# --- Sidebar controls ---
language = st.sidebar.selectbox("Select Language / ಭಾಷೆ ಆಯ್ಕೆಮಾಡಿ", ["English", "ಕನ್ನಡ"])
view_metrics = st.sidebar.button("View Metrics")

# Determine language code for Q&A generation page
lang_code = "en" if language == "English" else "kn"
stop_words = STOP_WORDS_EN if lang_code == "en" else STOP_WORDS_KN

# --- Load models ---
@st.cache_resource(show_spinner=True)
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # English QG (T5-based) via Auto*
    tokenizer_en = AutoTokenizer.from_pretrained("valhalla/t5-small-qg-hl")
    model_en = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qg-hl").to(device)

    # Kannada QG (ai4bharat) via Auto*
    tokenizer_kn = AutoTokenizer.from_pretrained(
        "ai4bharat/MultiIndicQuestionGenerationSS",
        do_lower_case=False, use_fast=False, keep_accents=True
    )
    model_kn = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/MultiIndicQuestionGenerationSS").to(device)

    return tokenizer_en, model_en, tokenizer_kn, model_kn

tokenizer_en, model_en, tokenizer_kn, model_kn = load_models()

def safe_word_tokenize(text, lang="en"):
    if lang == "en":
        return word_tokenize(text)
    else:
        # Basic split for non-English (Kannada)
        return text.split()

# --- Helper functions ---
def generate_wh_question(sentence, lang_code):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if lang_code == "en":
        prompt = f"generate question: {sentence}"
        inputs = tokenizer_en.encode(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        outputs = model_en.generate(inputs, max_length=64, num_beams=4, early_stopping=True)
        question = tokenizer_en.decode(outputs[0], skip_special_tokens=True)
    else:
        prompt = f"question: {sentence}"
        inputs = tokenizer_kn.encode(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        outputs = model_kn.generate(inputs, max_length=64, num_beams=4, early_stopping=True)
        question = tokenizer_kn.decode(outputs[0], skip_special_tokens=True)
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
    if len(words) >= 3:
        fake_options = random.sample(words, 3)
    else:
        fake_options = words.copy()
        # Language-specific dummy options
        dummy_options = ["Apple", "River", "Book", "Tree"] if lang_code == "en" else ["ಸೇಬು", "ನದಿ", "ಪುಸ್ತಕ", "ಮರ"]
        fake_options.extend(dummy_options[:3-len(fake_options)])
    options = [answer] + fake_options
    random.shuffle(options)
    question = sentence.replace(answer, "_____", 1)
    return question, options, answer

def generate_matching(sentences):
    left, right = [], []
    for sent in sentences:
        tokens = safe_word_tokenize(sent, "en" if lang_code=="en" else "kn")
        if lang_code == "en":
            tagged = pos_tag(tokens, lang="eng")
            nouns = [w for w, pos in tagged if pos.startswith("NN")]
        else:
            # simple KN heuristic: pick longer tokens as “nouns”
            nouns = [t for t in tokens if len(t) >= 4]
        if nouns:
            left.append(sent)
            right.append(random.choice(nouns))
    if not left:
        return []
    shuffled_right = right.copy()
    random.shuffle(shuffled_right)
    return [{"left": l, "right": r} for l, r in zip(left, shuffled_right)]

def simple_sent_tokenize(text):
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return [s for s in sentences if s]

def generate_questions_from_paragraph(paragraph, max_wh=5):
    sentences = simple_sent_tokenize(paragraph)
    questions = {"WH": [], "TrueFalse": [], "FillBlank": [], "MCQ": [], "Matching": []}

    for idx, sent in enumerate(sentences):
        if idx < max_wh:
            questions["WH"].append({"question": generate_wh_question(sent, lang_code), "answer": sent})
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

# --- Evaluation metrics for WH questions ---
def compute_wh_metrics(paragraphs, lang, tokenizer, model):
    lang_short = "en" if lang == "English" else "kn"
    references, hypotheses = [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for para in paragraphs:
        sentences = simple_sent_tokenize(para)
        for sent in sentences[:5]:
            prompt = f"generate question: {sent}" if lang=="English" else f"question: {sent}"
            inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
            outputs = model.generate(inputs, max_length=64, num_beams=4, early_stopping=True)
            question = tokenizer.decode(outputs[0], skip_special_tokens=True)
            hypotheses.append(safe_word_tokenize(question, lang_short))
            references.append([safe_word_tokenize(sent, lang_short)])

    bleu = corpus_bleu(references, hypotheses)
    sacre = sacrebleu.corpus_bleu([" ".join(h) for h in hypotheses], [[" ".join(r[0]) for r in references]])
    return {"BLEU": round(bleu, 4), "SacreBLEU": round(sacre.score, 4)}

# --- Sample paragraphs for evaluation ---
sample_paragraphs_en = [
    "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.",
    "The Nile is the longest river in Africa and has been essential to Egyptian civilization.",
    "Kannada is a Dravidian language spoken predominantly in the state of Karnataka, India."
]
sample_paragraphs_kn = [
    "ಅಲ್ಬರ್ಟ್ ಐನ್ಸ್ಟೈನ್ ಜರ್ಮನಿ ಜನಿಸಿದ ಸಿದ್ಧಾಂತ ಭೌತಶಾಸ್ತ್ರಜ್ಞರಾಗಿದ್ದರು, ಅವರು ಆಪೇಕ್ಷಾತ್ಮಕತೆಯ ಸಿದ್ಧಾಂತವನ್ನು ಅಭಿವೃದ್ಧಿಪಡಿಸಿದರು.",
    "ನೀಲ್ ನದಿ ಆಫ್ರಿಕಾದ ಉದ್ದನೆಯ ನದಿಯಾಗಿದ್ದು, ಈಜಿಪ್ಟ್ ನಾಗರಿಕತೆಗೆ ಅಗತ್ಯವಾಯಿತು.",
    "ಕನ್ನಡವು ಭಾರತದ ಕರ್ನಾಟಕ ರಾಜ್ಯದಲ್ಲಿ ಪ್ರಧಾನವಾಗಿ ಮಾತನಾಡುವ ದ್ರಾವಿಡ ಭಾಷೆಯಾಗಿದ್ದು."
]

# --- Streamlit pages ---
if view_metrics:
    st.title("Performance Metrics")
    st.markdown("Metrics are computed separately for English and Kannada models.")

    metrics_en = compute_wh_metrics(sample_paragraphs_en, "English", tokenizer_en, model_en)
    metrics_kn = compute_wh_metrics(sample_paragraphs_kn, "Kannada", tokenizer_kn, model_kn)

    st.subheader("Metrics for English")
    st.write(metrics_en)
    st.subheader("Metrics for Kannada")
    st.write(metrics_kn)

else:
    st.title(UI_TEXT[lang_code]["title"])
    st.write(UI_TEXT[lang_code]["desc"])
    paragraph = st.text_area(UI_TEXT[lang_code]["input"], height=200)
    if st.button(UI_TEXT[lang_code]["generate_btn"]):
        if not paragraph.strip():
            st.warning(UI_TEXT[lang_code]["warning"])
        else:
            q_data = generate_questions_from_paragraph(paragraph)
            st.subheader("Generated Questions")
            for qtype, qlist in q_data.items():
                st.markdown(f"### {qtype}")
                for idx, q in enumerate(qlist):
                    if qtype=="MCQ":
                        st.write(f"{idx+1}. {q['question']}")
                        st.write(f"Options: {q['options']}")
                        st.write(f"Answer: {q['answer']}")
                    elif qtype=="Matching":
                        st.write(f"{idx+1}. {q}")
                    else:
                        st.write(f"{idx+1}. {q['question']} → {q['answer']}")
            # CSV download
            rows = []
            for qtype, qlist in q_data.items():
                for q in qlist:
                    row = {"Type": qtype}
                    row.update(q)
                    rows.append(row)
            df = pd.DataFrame(rows)
            st.download_button(UI_TEXT[lang_code]["download"], df.to_csv(index=False), file_name="questions.csv")
