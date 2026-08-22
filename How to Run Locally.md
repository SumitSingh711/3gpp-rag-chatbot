# How to Run Locally

## Live Demo

🔗 **Live Demo:** [3GPP RAG Chatbot](https://3gpp-rag-chatbot-web.streamlit.app/)

---

## How to Run Locally with Custom 3GPP PDF

### 1. Go to the project folder

```bash
cd 3gpp-rag-chatbot
```

### 2. Automatic Environment Creation and Library Installation

Run the `setup.bat` file.

This will automatically create the required environment and install the project dependencies.(P.S. This will take some time around 5 to 10 minutes)

### 3. Add a New 3GPP PDF or use fully new pdf's (Optional)

To use a custom 3GPP specification:

1. Convert the specification file to PDF and place it inside:

```text
data/raw/
```

> The preprocessing pipeline is designed specifically for 3GPP-format PDF documents and automatically extracts the required metadata.

2. Run the pipeline after activating the virtual environment from terminal:

```bash
.venv\scripts\activate

python src/pipeline.py
```

The pipeline is **incremental**, so only newly added PDF files will be processed.

### 4. Run the Streamlit UI

Start the application with:

```bash
streamlit run src/app.py
```

The Streamlit application will start locally and provide a URL to access the chatbot in your browser.

---

## Evaluation

Run the hallucination evaluation script:

```bash
python eval/hallucination_eval.py
```

The evaluation reports:

- Answer rate on **in-corpus questions**
- Correct refusal rate on **out-of-corpus questions**
- Number of sentences flagged as **unsupported by the verifier**

To add or modify evaluation questions, update:

```text
eval/eval_set.jsonl
```

---
