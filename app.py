# app.py
# Streamlit bilingual question generator (English/Kannada) with Metrics page
import streamlit as st
st.set_page_config(page_title="Automatic Q&A Generator", layout="wide")

import torch
import random
import re
import pandas as pd

# HF imports
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForSeq2SeqLM
from transformers.models.auto import AutoTokenizer

# Metrics + tokenization helpers
from nltk.translate.bleu_score import corpus_bleu
import sacrebleu
from nltk.tokenize import word_tokenize, sent_tokenize
import os
import nltk
from nltk import pos_tag

# ---------- NLTK setup ----------
def _ensure_nltk_data():
    dl_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
    os.makedirs(dl_dir, exist_ok=True)
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
            nltk.download(pkg, download_dir=dl_dir, quiet=True)

_ensure_nltk_data()

# ---------- UI: Colors (hover -> blue) ----------
st.markdown("""
<style>
/* Make all buttons hover blue */
button[kind="primary"]:hover, button:hover {
  background-color: #1e88e5 !important;
  border-color: #1565c0 !important;
  color: white !important;
}
/* Download button hover */
.stDownloadButton > button:hover {
  background-color: #1e88e5 !important;
  border-color: #1565c0 !important;
}
/* Expander title hover */
details:hover > summary, details > summary:hover {
  color: #1e88e5 !important;
}
/* Links hover */
a:hover { color: #1e88e5 !important; }
</style>
""", unsafe_allow_html=True)

# ---------- UI text ----------
UI_TEXT = {
    "en": {
        "title": "Automatic Q&A Generator",
        "desc": "Enter a paragraph below, and the app will generate multiple types of questions.",
        "input": "Paste your paragraph here:",
        "sliders_title": "How many questions per type?",
        "wh": "WH questions",
        "tf": "True/False",
        "fb": "Fill in the Blank",
        "mcq": "MCQ",
        "match": "Matching pairs",
        "generate_btn": "Generate Questions",
        "warning": "Please enter a paragraph!",
        "download": "Download as CSV"
    },
    "kn": {
        "title": "ಸ್ವಯಂಚಾಲಿತ ಪ್ರಶ್ನೆ-ಉತ್ತರ ತಯಾರಕ",
        "desc": "ಕೆಳಗಿನ ಪ್ಯಾರಾಗ್ರಾಫ್ ಅನ್ನು ನಮೂದಿಸಿ, ಅಪ್ಲಿಕೇಶನ್ ವಿವಿಧ ರೀತಿಯ ಪ್ರಶ್ನೆಗಳನ್ನು ತಯಾರಿಸುತ್ತದೆ.",
        "input": "ನಿಮ್ಮ ಪ್ಯಾರಾಗ್ರಾಫ್ ಅನ್ನು ಇಲ್ಲಿ ಹಾಕಿ:",
        "sliders_title": "ಪ್ರಕಾರಪ್ರತಿ ಎಷ್ಟು ಪ್ರಶ್ನೆಗಳು?",
        "wh": "WH ಪ್ರಶ್ನೆಗಳು",
        "tf": "ಸತ್ಯ/ಅಸತ್ಯ",
        "fb": "ಖಾಲಿ ಜಾಗ ತುಂಬಿಸಿ",
        "mcq": "ಬಹು ಆಯ್ಕೆ (MCQ)",
        "match": "ಹೊಂದಾಣಿಕೆ ಜೋಡಿಗಳು",
        "generate_btn": "ಪ್ರಶ್ನೆಗಳನ್ನು ತಯಾರಿಸಿ",
        "warning": "ದಯವಿಟ್ಟು ಪ್ಯಾರಾಗ್ರಾಫ್ ನಮೂದಿಸಿ!",
        "download": "CSV ಆಗಿ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ"
    }
}

# ---------- Stopwords ----------
STOP_WORDS_EN = set("""
a an the and or but if while with is are was were has have had of in on for
to from by as at that this it
""".split())

STOP_WORDS_KN = set([
    "ಆಗ", "ಅವರು", "ಅನ್ನು", "ಅಲ್ಲಿ", "ಇದ್ದ", "ಇದೆ", "ಮತ್ತು", "ಅದನ್ನು",
    "ಅಥವಾ", "ಆದರೆ", "ಇದೆ", "ಇದ್ದವು", "ಇದನ್ನು", "ಅವರು"
])

# ---------- Sidebar controls ----------
language = st.sidebar.selectbox("Select Language / ಭಾಷೆ ಆಯ್ಕೆಮಾಡಿ", ["English", "ಕನ್ನಡ"])
view_metrics = st.sidebar.button("View Metrics")
lang_code = "en" if language == "English" else "kn"
stop_words = STOP_WORDS_EN if lang_code == "en" else STOP_WORDS_KN

# ---------- Load models ----------
@st.cache_resource(show_spinner=True)
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer_en = T5Tokenizer.from_pretrained("valhalla/t5-small-qg-hl")
    model_en = T5ForConditionalGeneration.from_pretrained("valhalla/t5-small-qg-hl").to(device)

    tokenizer_kn = AutoTokenizer.from_pretrained(
        "ai4bharat/MultiIndicQuestionGenerationSS",
        do_lower_case=False, use_fast=False, keep_accents=True
    )
    model_kn = AutoModelForSeq2SeqLM.from_pretrained(
        "ai4bharat/MultiIndicQuestionGenerationSS"
    ).to(device)

    return tokenizer_en, model_en, tokenizer_kn, model_kn

tokenizer_en, model_en, tokenizer_kn, model_kn = load_models()

# ---------- Sanitizers ----------
_FANCY_QUOTES = {
    "“":"\"", "”":"\"", "‘":"'", "’":"'", "—":"-", "–":"-", "…":"..."
}
_META_PAT = re.compile(
    r"(starts with|begin(s)? with|first question|based on|about:|prompt|instruction|prefix)",
    re.I
)

def _normalize(s: str) -> str:
    for k,v in _FANCY_QUOTES.items():
        s = s.replace(k,v)
    s = re.sub(r"\s+", " ", s.strip())
    return s

def _strip_meta(s: str) -> str:
    s = _normalize(s)
    s = _META_PAT.sub("", s)
    s = re.sub(r'["“”]+', "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _lex_overlap(a: str, b: str, stopset) -> float:
    A = {w.lower() for w in re.findall(r"[A-Za-z]+", _normalize(a)) if w.lower() not in stopset}
    B = {w.lower() for w in re.findall(r"[A-Za-z]+", _normalize(b)) if w.lower() not in stopset}
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

# ---------- Tokenization ----------
def safe_word_tokenize(text, lang="en"):
    return word_tokenize(text) if lang == "en" else text.split()

def simple_sent_tokenize(text):
    text = re.sub(r"\s+", " ", text.strip())
    try:
        return [s for s in sent_tokenize(text) if s.strip()]
    except Exception:
        return [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

# ===== English WH selection & prefix building =====

# IMPORTANT: remove generic 'planet' to avoid false astronomy triggers
_DOMAIN_LEX = {
    "planet":  r"\b(Mercury|Venus|Earth|Mars|Jupiter|Saturn|Uranus|Neptune)\b",
    "city":    r"\b(city|cities|Harappa|Mohenjo-?Daro|Babylon|Ur|London|Paris|Delhi|Bengaluru)\b",
    "river":   r"\b(river|Nile|Indus|Amazon|Ganga|Ganges|Cauvery|Kaveri)\b",
    "language":r"\b(language|dialect|Kannada|English|Hindi|Tamil|Telugu)\b",
    "country": r"\b(country|nation|state|India|Pakistan|Brazil|USA|United States|China)\b",
}

_TIME_HINT = r"\b(BC|BCE|AD|CE|century|year|years|month|months|week|weeks|day|days|season|era|age|aged|born|died|around \d{3,4}|\b\d{3,4}\b)\b"
_REASON_HINT = r"\b(because|due to|therefore|so that|reason|purpose|cause|caused)\b"
_PLACE_HINT = r"\b(located|location|region|valley|continent|ocean|country|state|city|capital|river|mountain|coast)\b"
_SUPERLATIVE_HINT = r"\b(closest|nearest|farthest|largest|smallest|biggest|only|main|primary|principal|first|most|least|highest|lowest)\b"
_QUANT_HINT = r"\b(\d+|many|much|several|few|number of|amount|percent|percentage|population|millions)\b"

def _detect_domain(sent: str) -> str | None:
    for name, pat in _DOMAIN_LEX.items():
        if re.search(pat, sent, re.I):
            return name
    return None

def _extract_head_noun(tokens_with_pos):
    for tag_pref in ("NNP", "NNPS", "NN", "NNS"):
        for w, t in tokens_with_pos:
            if t == tag_pref and re.sub(r"\W+", "", w):
                return w
    return None

def choose_wh_and_prefix(sentence: str):
    s = " " + sentence.strip() + " "
    domain = _detect_domain(sentence)
    tokens = safe_word_tokenize(sentence, "en")
    tagged = pos_tag(tokens, lang="eng")
    head = _extract_head_noun(tagged)

    if re.search(_REASON_HINT, s, re.I): return "why", "Why"
    if re.search(_TIME_HINT, s, re.I):   return "when", "When"
    if re.search(_QUANT_HINT, s, re.I):  return "how many", "How many"

    if re.search(_SUPERLATIVE_HINT, s, re.I) or domain:
        noun = None
        if domain in ("planet", "city", "river", "language", "country"):
            noun = domain
        elif head and head[0].isupper():
            noun = "entity"
        elif head:
            noun = head.lower()
        noun = noun or "item"
        return "which", f"Which {noun}"

    if re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", sentence): return "who", "Who"
    if re.search(_PLACE_HINT, s, re.I):  return "where", "Where"
    if head: return "which", f"Which {head.lower()}"
    return "what", "What"

def _polish_question(q: str) -> str:
    q = _normalize(q)
    q = q[0].upper() + q[1:] if q else q
    if not q.endswith("?"):
        q += "?"
    q = q.replace(" ,", ",").replace(" .", ".").replace(" ?", "?")
    q = re.sub(r"\b[Ii]s is\b", "is", q)
    return q

def _enforce_prefix(q: str, prefix: str) -> str:
    q_clean = _strip_meta(q)
    if not q_clean.lower().startswith(prefix.lower()):
        if re.match(r"^(who|what|when|where|why|which|how)(\s+many)?\b", q_clean, re.I):
            q_clean = re.sub(r"^(who|what|when|where|why|which|how)(\s+many)?",
                             prefix, q_clean, flags=re.I)
        else:
            q_clean = f"{prefix} {q_clean}"
    return _polish_question(q_clean)

def _is_low_quality(q: str, sentence: str, stopset) -> bool:
    if len(q.split()) < 3: return True
    if not q.endswith("?"): return True
    if re.search(r"\b(where|what)\s+[A-Z][a-z]+\s+is\b", q, flags=re.I): return True
    if _lex_overlap(q, sentence, stopset) < 0.15: return True
    return False

def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

# ===== Kannada quality helpers =====
_KN_TIME = r"(ಶತಮಾನ|ವರ್ಷ|ತಿಂಗಳು|ವಾರ|ದಿನ|ಯುಗ|ಕಾಲ|ಆಯುಷ್ಯ|ಜನನ|ಮರಣ|\bಇಸಾಪೂರ್ವ\b|\bಕ್ರಿ\.ಶ\b|\bಕ್ರಿ\.ಪೂ\b|\b\d{3,4}\b)"
_KN_PLACE = r"(ನಗರ|ಗ್ರಾಮ|ಪ್ರದೇಶ|ರಾಜ್ಯ|ದೇಶ|ಸ್ಥಳ|ನದಿ|ಪರ್ವತ|ತೀರ|ಭೂಖಂಡ|ಖಂಡ|ಸಮುದ್ರ|ಸ್ಥಿತ|ವಿರುವ)"
_KN_REASON = r"(ಯಾಕೆಂದರೆ|ಕಾರಣ|ಹಾಗಾಗಿ|ಆದ್ದರಿಂದ|ಉದ್ದೇಶ|ಕಾರಣವಾಗಿ)"
_KN_QUANT = r"(ಎಷ್ಟು|ಸಂಖ್ಯೆ|ಪ್ರಮಾಣ|\b\d+\b|ಲಕ್ಷ|ಕೋಟಿ)"
_KN_PERSON = r"(ವ್ಯಕ್ತಿ|ರಾಜ|ನಾಯಕ|ವಿಜ್ಞಾನಿ|ಲೇಖಕ|ಸಂಗೀತಗಾರ|ಸನ್ನ್ಯಾಸಿ|ಕವಿ)"

_KN_DOMAIN = [
    ("ಗ್ರಹ", r"(ಬುಧ|ಶುಕ್ರ|ಭೂಮಿ|ಮಂಗಳ|ಗುರು|ಶನಿ|ಯುರೇನಸ್|ನೆಪ್ಟ್ಯೂನ್)"),
    ("ನಗರ", r"(ನಗರ|ಹರಪ್ಪಾ|ಮೋಹೆಂಜೋದಾರೋ|ಬೆಂಗಳೂರ|ಮೈಸೂರು|ಮಂಗಳೂರು)"),
    ("ನದಿ", r"(ನದಿ|ಗಂಗಾ|ಕಾವೇರಿ|ಸಿಂಧು|ನೈಲ್|ಅಮೆಜಾನ್)"),
    ("ಭಾಷೆ", r"(ಭಾಷೆ|ಕನ್ನಡ|ಹಿಂದಿ|ತಮಿಳು|ತೆಲುಗು|ಇಂಗ್ಲಿಷ್)"),
    ("ದೇಶ", r"(ದೇಶ|ಭಾರತ|ಪಾಕಿಸ್ತಾನ|ಅಮೇರಿಕಾ|ಚೀನಾ|ಬ್ರೆಝಿಲ್)"),
]

def _kn_detect_domain(s: str):
    for noun, pat in _KN_DOMAIN:
        if re.search(pat, s, flags=re.I):
            return noun
    return None

def choose_wh_and_prefix_kn(sentence: str):
    s = " " + sentence.strip() + " "
    if re.search(_KN_REASON, s): return "ಏಕೆ", "ಏಕೆ"
    if re.search(_KN_TIME, s):   return "ಯಾವಾಗ", "ಯಾವಾಗ"
    if re.search(_KN_QUANT, s):  return "ಎಷ್ಟು", "ಎಷ್ಟು"
    dom = _kn_detect_domain(s)
    if dom:                      return "ಯಾವ", f"ಯಾವ {dom}"
    if re.search(_KN_PLACE, s):  return "ಎಲ್ಲಿ", "ಎಲ್ಲಿ"
    if re.search(_KN_PERSON, s): return "ಯಾರು", "ಯಾರು"
    return "ಏನು", "ಏನು"

def _polish_kn(q: str) -> str:
    q = _normalize(q)
    if not q.endswith("?"): q += "?"
    q = q.replace(" ,", ",").replace(" .", ".").replace(" ?", "?")
    return q

def _enforce_prefix_kn(q: str, prefix: str) -> str:
    qc = _strip_meta(q)
    starters = r"(ಯಾರು|ಏನು|ಯಾವಾಗ|ಎಲ್ಲಿ|ಏಕೆ|ಯಾವ|ಹೇಗೆ|ಎಷ್ಟು)"
    if not re.match("^" + starters, qc):
        qc = f"{prefix} {qc}"
    else:
        qc = re.sub("^" + starters, prefix, qc)
    return _polish_kn(qc)

# ---------- Question Generators ----------
def _clean_prompt_context(s: str) -> str:
    # keep context simple; avoid leaking meta phrases
    return _normalize(s)

def generate_wh_question(sentence, lang_code):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if lang_code == "en":
        wh, prefix = choose_wh_and_prefix(sentence)
        ctx = _clean_prompt_context(sentence)
        # Safer prompt: no "starts with ..." phrase
        prompt = f"context: {ctx}\nWrite one concise {wh} question about the context. Do not mention instructions."
        inputs = tokenizer_en.encode(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        outputs = model_en.generate(
            inputs, max_length=48, num_beams=4, do_sample=True, top_p=0.9, temperature=0.9, early_stopping=True
        )
        raw_q = tokenizer_en.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        q = _enforce_prefix(_strip_meta(raw_q), prefix)
        if _is_low_quality(q, sentence, STOP_WORDS_EN):
            # last-resort clean template anchored on a noun
            toks = safe_word_tokenize(sentence, "en")
            head = _extract_head_noun(pos_tag(toks, lang="eng")) or "item"
            if prefix.lower().startswith("which"):
                q = f"{prefix} {head.lower()} is being described in the passage?"
            elif prefix.lower().startswith("how many"):
                q = f"{prefix} {head.lower()} are mentioned?"
            else:
                q = prefix + "?"
            q = _polish_question(q)
        return q

    # Kannada
    wh_kn, prefix_kn = choose_wh_and_prefix_kn(sentence)
    ctx = _clean_prompt_context(sentence)
    prompt = f"ಸಂದರ್ಭ: {ctx}\nಸಂದರ್ಭದ ಬಗ್ಗೆ ಒಂದು ಸಂಕ್ಷಿಪ್ತ ಪ್ರಶ್ನೆ ಬರೆಯಿರಿ. ಸೂಚನೆಗಳನ್ನು ಉಲ್ಲೇಖಿಸಬೇಡಿ."
    inputs = tokenizer_kn.encode(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model_kn.generate(
        inputs, max_length=48, num_beams=4, do_sample=True, top_p=0.92, temperature=0.9, early_stopping=True
    )
    raw_q = tokenizer_kn.decode(outputs[0], skip_special_tokens=True)
    q = _enforce_prefix_kn(_strip_meta(raw_q), prefix_kn)
    # light quality check for Kannada: overlap with context
    if len(q.split()) < 3 or _lex_overlap(q, sentence, STOP_WORDS_KN) < 0.1:
        q = prefix_kn + "?"
    return q

_NUM_RE = re.compile(r"\b\d{1,4}\b")

def _find_proper_nouns(tokens_pos):
    return [w for w,t in tokens_pos if t in ("NNP","NNPS") and re.sub(r"\W+","",w)]

def _flip_number(text):
    def repl(m):
        n = int(m.group(0))
        delta = 1 if n < 50 else (10 if n < 1000 else 100)
        return str(n + delta)
    if _NUM_RE.search(text):
        return _NUM_RE.sub(repl, text, count=1), True
    return text, False

def _negate_sentence_once(text):
    patterns = [r"\bis\b", r"\bare\b", r"\bwas\b", r"\bwere\b", r"\bhas\b", r"\bhave\b"]
    for p in patterns:
        if re.search(p, text):
            return re.sub(p, lambda m: m.group(0) + " not", text, count=1), True
    return text, False

# Kannada True/False helpers
_KN_NEG_MAP = [
    (r"\bಇದೆ\b", "ಇಲ್ಲ"),
    (r"\bಇದ್ದಾರೆ\b", "ಇಲ್ಲ"),
    (r"\bಇರುತ್ತದೆ\b", "ಇರುವುದಿಲ್ಲ"),
    (r"\bಇರುವ\b", "ಇಲ್ಲದ"),
    (r"\bಆಗಿದೆ\b", "ಆಗಿಲ್ಲ"),
    (r"\bಸಂಭವಿಸಿದೆ\b", "ಸಂಭವಿಸಿಲ್ಲ"),
]
def _kn_negate_once(text: str):
    for pat, rep in _KN_NEG_MAP:
        if re.search(pat, text):
            return re.sub(pat, rep, text, count=1), True
    return text, False

def generate_true_false_kn(sentence, paragraph_context=None):
    if random.random() < 0.5:
        return sentence, "True"
    changed, ok = _flip_number(sentence)
    if ok: return _normalize(changed), "False"
    changed, ok = _kn_negate_once(sentence)
    if ok: return _normalize(changed), "False"
    return _normalize(sentence), "True"

def generate_true_false(sentence, paragraph_context=None):
    if lang_code == "kn":
        return generate_true_false_kn(sentence, paragraph_context)
    if random.random() < 0.5:
        return _normalize(sentence), "True"
    changed, ok = _flip_number(sentence)
    if ok: return _normalize(changed), "False"
    changed, ok = _negate_sentence_once(sentence)
    if ok: return _normalize(changed), "False"
    if paragraph_context:
        toks = safe_word_tokenize(sentence, "en")
        tagged = pos_tag(toks, lang="eng")
        pns = _find_proper_nouns(tagged)
        if pns:
            ctx_tokens = safe_word_tokenize(paragraph_context, "en")
            ctx_pns = _find_proper_nouns(pos_tag(ctx_tokens, lang="eng"))
            pool = [p for p in set(ctx_pns) if p not in set(pns)]
            if pool:
                target = random.choice(pns)
                subst  = random.choice(pool)
                return _normalize(re.sub(rf"\b{re.escape(target)}\b", subst, sentence, count=1)), "False"
    return _normalize(sentence), "True"

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-’']+")

def _content_words(sentence, stopset):
    toks = safe_word_tokenize(sentence, "en")
    tags = pos_tag(toks, lang="eng")
    candidates = [w for w,t in tags if t in ("NNP","NNPS","NN","NNS")]
    if not candidates:
        candidates = [w for w in toks if len(w) > 3 and w.lower() not in stopset and _WORD_RE.fullmatch(w)]
    return candidates

def _content_words_kn(sentence, stopset):
    toks = [t for t in re.split(r"\W+", sentence) if t]
    toks = [t for t in toks if len(t) >= 4 and t not in stopset]
    return toks

def _clean_option(tok: str) -> str:
    tok = _normalize(tok)
    tok = re.sub(r'^[\'"“”]+|[\'"“”]+$', "", tok)
    tok = re.sub(r"^[^\w]+|[^\w]+$", "", tok)
    return tok

def generate_fill_blank(sentence):
    if lang_code == "kn":
        cands = _content_words_kn(sentence, STOP_WORDS_KN)
        if not cands: return None, None
        answer = random.choice(list(set(cands)))
        stem = re.sub(rf"\b{re.escape(answer)}\b", "_____", sentence, count=1)
        return _normalize(stem), _normalize(answer)
    cands = _content_words(sentence, stop_words)
    if not cands: return None, None
    answer = random.choice(list(set(cands)))
    stem = re.sub(rf"\b{re.escape(answer)}\b", "_____", sentence, count=1)
    return _normalize(stem), _clean_option(answer)

def _is_number(tok): return bool(_NUM_RE.fullmatch(tok))

def _nearby_numbers(n, k=3):
    n = int(n)
    deltas = [1,2,5] if n < 50 else [5,10,20]
    opts = list({str(n+d) for d in deltas} | {str(max(1, n-d)) for d in deltas})
    random.shuffle(opts)
    return opts[:k]

def _collect_pos_pools(paragraph):
    toks = safe_word_tokenize(paragraph, "en")
    tags = pos_tag(toks, lang="eng")
    proper = [w for w,t in tags if t in ("NNP","NNPS")]
    common = [w for w,t in tags if t in ("NN","NNS") and w.lower() not in STOP_WORDS_EN]
    return list(set(proper)), list(set(common))

def _generate_mcq_english(sentence, paragraph_context=None):
    cands = _content_words(sentence, stop_words)
    if not cands: return None, None, None
    answer = _clean_option(random.choice(list(set(cands))))
    if not answer: return None, None, None

    if _is_number(answer):
        distractors = _nearby_numbers(answer, k=3)
    else:
        proper_pool, common_pool = _collect_pos_pools(paragraph_context or sentence)
        if answer in proper_pool:
            pool = [ _clean_option(w) for w in proper_pool if _clean_option(w) and _clean_option(w) != answer ]
        else:
            pool = [ _clean_option(w) for w in common_pool if _clean_option(w).lower() != answer.lower() ]
        pool = [p for p in pool if p and len(p) >= 2]
        distractors = random.sample(pool, min(3, len(pool))) if pool else []

    while len(distractors) < 3:
        fillers = ["Amazon", "Brazil", "forest", "climate", "biodiversity", "river"]
        fillers = [f for f in fillers if f.lower() != answer.lower() and f not in distractors]
        if not fillers: break
        distractors.append(fillers.pop(0))

    options = [answer] + distractors[:3]
    options = list(dict.fromkeys([_clean_option(o) for o in options if _clean_option(o)]))
    random.shuffle(options)
    if len(options) < 4:  # ensure 4 choices when possible
        pad = ["region", "ecosystem", "species", "country"]
        for p in pad:
            if p not in options: options.append(p)
            if len(options) >= 4: break
    stem = re.sub(rf"\b{re.escape(answer)}\b", "_____", sentence, count=1)
    return _normalize(stem), options[:4], answer

def generate_mcq(sentence, paragraph_context=None):
    if lang_code == "kn":
        cands = _content_words_kn(sentence, STOP_WORDS_KN)
        if not cands: return None, None, None
        answer = random.choice(list(set(cands)))
        para_cands = _content_words_kn(paragraph_context or sentence, STOP_WORDS_KN)
        pool = [w for w in set(para_cands) if w != answer and len(w) >= 2]
        distractors = random.sample(pool, min(3, len(pool))) if pool else []
        fillers = ["ಅಮೆಜಾನ್", "ಬ್ರೆಝಿಲ್", "ಅರಣ್ಯ", "ಜೀವ ವೈವಿಧ್ಯ", "ಪ್ರದೇಶ", "ನದಿ"]
        for f in fillers:
            if len(distractors) >= 3: break
            if f != answer and f not in distractors:
                distractors.append(f)
        options = [answer] + distractors[:3]
        options = list(dict.fromkeys([_normalize(o) for o in options if _normalize(o)]))
        random.shuffle(options)
        stem = re.sub(rf"\b{re.escape(answer)}\b", "_____", sentence, count=1)
        return _normalize(stem), options[:4], _normalize(answer)

    return _generate_mcq_english(sentence, paragraph_context)

def generate_matching(sentences, max_pairs=5):
    pairs = []
    for s in sentences:
        if lang_code == "en":
            toks = safe_word_tokenize(s, "en")
            tags = pos_tag(toks, lang="eng")
            # prefer proper nouns and numbers/dates
            nouns = [w for w,t in tags if t in ("NNP","NNPS")]
            if not nouns:
                nouns = [w for w,t in tags if t in ("CD",)]
            # fallback: meaningful common nouns (not generic)
            if not nouns:
                nouns = [w for w,t in tags if t in ("NN","NNS") and w.lower() not in {"majority","people","thing","area"}]
        else:
            toks = [t for t in re.split(r"\W+", s) if t]
            nouns = [t for t in toks if len(t) >= 4]

        if nouns:
            term = random.choice(nouns)
            right = s if len(s) <= 140 else s[:137] + "…"
            term = _clean_option(term) if lang_code == "en" else _normalize(term)
            if term and term.lower() not in {"majority","question","context"}:
                pairs.append({"left": term, "right": _normalize(right)})

    random.shuffle(pairs)
    pairs = pairs[:max_pairs]
    rights = [p["right"] for p in pairs]
    random.shuffle(rights)
    return [{"left": p["left"], "right": r} for p, r in zip(pairs, rights)]

def _dedupe_wh(wh_list, sim_threshold=0.6):
    kept = []
    for q in wh_list:
        if all(jaccard(q["question"], k["question"]) < sim_threshold for k in kept):
            kept.append(q)
    return kept

# Accept per-type limits
def generate_questions_from_paragraph(paragraph, n_wh=5, n_tf=10, n_fb=10, n_mcq=10, n_match=5):
    paragraph = _normalize(paragraph)
    sentences = simple_sent_tokenize(paragraph)

    questions = {"WH": [], "TrueFalse": [], "FillBlank": [], "MCQ": [], "Matching": []}

    for sent in sentences:
        if len(questions["WH"]) >= n_wh: break
        q = generate_wh_question(sent, lang_code)
        questions["WH"].append({"question": q, "answer": _normalize(sent)})
    if questions["WH"]:
        questions["WH"] = _dedupe_wh(questions["WH"])[:n_wh]

    for sent in sentences:
        if len(questions["TrueFalse"]) >= n_tf: break
        tf_q, tf_a = generate_true_false(sent, paragraph_context=paragraph)
        questions["TrueFalse"].append({"question": tf_q, "answer": tf_a})

    for sent in sentences:
        if len(questions["FillBlank"]) >= n_fb: break
        fb_q, fb_a = generate_fill_blank(sent)
        if fb_q:
            questions["FillBlank"].append({"question": fb_q, "answer": fb_a})

    for sent in sentences:
        if len(questions["MCQ"]) >= n_mcq: break
        mcq_q, options, mcq_a = generate_mcq(sent, paragraph_context=paragraph)
        if mcq_q and options:
            questions["MCQ"].append({"question": mcq_q, "options": options, "answer": mcq_a})

    questions["Matching"] = generate_matching(sentences, max_pairs=n_match)

    return questions

# ---------- Metrics ----------
def compute_wh_metrics(paragraphs, lang, tokenizer, model):
    lang_short = "en" if lang == "English" else "kn"
    references, hypotheses = [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for para in paragraphs:
        sentences = simple_sent_tokenize(para)
        for sent in sentences[:5]:
            prompt = f"generate question: {sent}" if lang == "English" else f"question: {sent}"
            inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
            outputs = model.generate(inputs, max_length=64, num_beams=4, early_stopping=True)
            question = tokenizer.decode(outputs[0], skip_special_tokens=True)
            hypotheses.append(safe_word_tokenize(question, lang_short))
            references.append([safe_word_tokenize(sent, lang_short)])
    bleu = corpus_bleu(references, hypotheses)
    sacre = sacrebleu.corpus_bleu([" ".join(h) for h in hypotheses],
                                  [[" ".join(r[0]) for r in references]])
    return {"BLEU": round(bleu, 4), "SacreBLEU": round(sacre.score, 4)}

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

# ---------- UI ----------
st.title(UI_TEXT[lang_code]["title"])
st.markdown(UI_TEXT[lang_code]["desc"])

if view_metrics:
    st.header("Performance Metrics")
    st.caption("Metrics are computed separately for English and Kannada models.")
    with st.spinner("Computing metrics..."):
        metrics_en = compute_wh_metrics(sample_paragraphs_en, "English", tokenizer_en, model_en)
        metrics_kn = compute_wh_metrics(sample_paragraphs_kn, "Kannada", tokenizer_kn, model_kn)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("English")
        st.json(metrics_en)
    with col2:
        st.subheader("Kannada")
        st.json(metrics_kn)
else:
    paragraph_input = st.text_area(UI_TEXT[lang_code]["input"], height=200)

    # Per-type sliders (appear for both languages)
    st.markdown(f"**{UI_TEXT[lang_code]['sliders_title']}**")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        n_wh   = st.slider(UI_TEXT[lang_code]["wh"],    min_value=0, max_value=5, value=3,  step=1)
    with c2:
        n_tf   = st.slider(UI_TEXT[lang_code]["tf"],    min_value=0, max_value=5, value=3,  step=1)
    with c3:
        n_fb   = st.slider(UI_TEXT[lang_code]["fb"],    min_value=0, max_value=5, value=3,  step=1)
    with c4:
        n_mcq  = st.slider(UI_TEXT[lang_code]["mcq"],   min_value=0, max_value=5, value=3,  step=1)
    with c5:
        n_match= st.slider(UI_TEXT[lang_code]["match"], min_value=0, max_value=5, value=2,  step=1)

    if st.button(UI_TEXT[lang_code]["generate_btn"]):
        if paragraph_input.strip() == "":
            st.warning(UI_TEXT[lang_code]["warning"])
        else:
            with st.spinner("Generating questions..."):
                qa_pairs = generate_questions_from_paragraph(
                    paragraph_input,
                    n_wh=n_wh, n_tf=n_tf, n_fb=n_fb, n_mcq=n_mcq, n_match=n_match
                )

            for qtype, qlist in qa_pairs.items():
                with st.expander(f"{qtype} Questions ({len(qlist)})", expanded=False):
                    for idx, q in enumerate(qlist, 1):
                        if qtype == "MCQ":
                            st.markdown(f"**Q{idx}:** {q['question']}")
                            for opt in q['options']:
                                st.markdown(f"- {opt}")
                            st.markdown(f"**Answer:** {q['answer']}")
                        elif qtype == "Matching":
                            st.markdown(f"**Pair {idx}:** Left → {q['left']} | Right → {q['right']}")
                        else:
                            st.markdown(f"**Q{idx}:** {q['question']}")
                            st.markdown(f"**Answer:** {q['answer']}")

            # CSV download
            rows = []
            for qtype, qlist in qa_pairs.items():
                if qtype != "Matching":
                    for q in qlist:
                        row = {"Type": qtype, "Question": q['question'], "Answer": q['answer']}
                        if qtype == "MCQ":
                            row["Options"] = ", ".join(q['options'])
                        rows.append(row)
                else:
                    for q in qlist:
                        rows.append({"Type": "Matching", "Question": q['left'], "Answer": q['right'], "Options": ""})
            df = pd.DataFrame(rows)
            st.download_button(
                label=UI_TEXT[lang_code]["download"],
                data=df.to_csv(index=False),
                file_name="generated_questions.csv",
                mime="text/csv"
            )
