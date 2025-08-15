# Automatic-Question-Answer-Generation
This is a Streamlit-based web application that automatically generates multiple types of questions from a given paragraph in **English** or **Kannada**. It supports WH questions, True/False, Fill-in-the-Blank, MCQs, and Matching questions, along with evaluation metrics.

---

## Features

- **Bilingual Q&A Generation**: Supports English and Kannada.
- **Multiple Question Types**:
  - WH questions (Who, What, When, Where, Why, How)
  - True/False questions
  - Fill-in-the-Blank
  - Multiple Choice Questions (MCQ)
  - Matching questions
- **Evaluation Metrics**: BLEU and SacreBLEU scores.
- **Download Questions**: Export generated questions as a CSV file.

---

## Installation

Local deployment-

1. Clone the repository and install the required dependencies:
```bash
git clone https://github.com/elainejpai-14/Automatic-Question-Answer-Generation.git
cd Automatic-Question-Answer-Generation
pip install -r requirements.txt
```

2. Run the application locally:
```bash
streamlit run app.py
```

---

## Usage

Once the app is running:<br>

1. Select the desired language (English or Kannada) from the sidebar.
2. Paste a paragraph into the provided text area.
3. Click on "Generate Questions" to see the generated questions.
4. Optionally, click on "View Metrics" to evaluate the performance of the question generation models.

---

## Models Used

- **English-** ```valhalla/t5-small-qg-hl``` for WH-question generation.
- **Kannada-** ```ai4bharat/MultiIndicQuestionGenerationSS``` for WH-question generation.

---

## Evaluation Metrics

The quality of the generated WH-questions is assessed using:

- BLEU: Measures the precision of n-grams between the generated and reference questions.

- SacreBLEU: A standardized version of BLEU for better reproducibility.

| Language  | BLEU Score | SacreBLEU Score |
|-----------|------------|----------------|
| English   | 0.3516     | 36.5355        |
| Kannada   | 0.4541     | 43.9352        |

Streamlit app deployed: https://automatic-question-answer-generation.streamlit.app/
