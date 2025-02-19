# **LLM Translation Evaluator**

A thorough framework for benchmarking translations produced by various Large Language Models (LLMs). This project provides:

- **BLEU** scoring
- **ROUGE** evaluation
- **Cosine similarity** (via embeddings)

so that users can assess the effectiveness of each model's translations against curated reference texts.

<br>

---

<br>

## **Table of Contents**
1. **Project Overview**
2. **Features**
3. **Project Structure**
4. **Installation**
5. **Usage**
6. **Configuration**
7. **Contributing**
8. **License**

<br>

---

<br>

## **Project Overview**

This repository aims to measure the translation quality of multiple LLMs (including OpenAI, Anthropic, Google, Perplexity, Deepseek, etc.) against reference translations. We calculate key metrics:

- **BLEU** – Checks n-gram overlaps.
- **ROUGE** – Emphasizes recall-based metrics.
- **Cosine Similarity** – Gauges closeness of embeddings for generated vs. reference translations.

Ultimately, the process yields a clear, side-by-side comparison of each model's performance.

<br>

---

<br>

## **Features**

- **Automated Translation** across various providers.
- **Robust Evaluation** leveraging BLEU, ROUGE, and embeddings.
- **Straightforward Customization** via environment variables or direct script edits.
- **Comprehensive Summaries** for easy model comparison.

<br>

---

<br>

## **Project Structure**

```
./
    benchmark.py
    llmclient.py
    embedding_scorer.py
    test_data.py
    evaluation_metrics.py
    main.py
    src/
        llm_translation_evaluator/
            __init__.py
```

- **`benchmark.py`**: Primary script for configuring and running translation benchmarks.
- **`llmclient.py`**: Integrates different LLM services.
- **`embedding_scorer.py`**: Computes embeddings and calculates cosine similarity.
- **`test_data.py`**: Houses sample text and reference translations.
- **`evaluation_metrics.py`**: Implements BLEU and ROUGE metrics.
- **`main.py`**: Core logic for the benchmarking routine, typically invoked by `benchmark.py`.
- **`src/llm_translation_evaluator/`**: Root package folder.

<br>

---

<br>

## **Installation**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/vncne/llm-language-eval.git
   ```

2. **Install Dependencies** (using Poetry):
   ```bash
   poetry install
   ```

3. **Set Environment Variables**:
   - Create a `.env` file with your API keys (OpenAI, Anthropic, Google, Perplexity, Deepseek).

<br>

---

<br>

## **Usage**

1. **Environment Variables**:
   In `.env`, specify:
   ```env
   OPENAI_API_KEY=<your-openai-key>
   ANTHROPIC_API_KEY=<your-anthropic-key>
   GEMINI_API_KEY=<your-google-genai-key>
   PERPLEXITY_API_KEY=<your-perplexity-key>
   DEEPOSEEK_API_KEY=<your-deepseek-key>
   ```

2. **Configure Models**:
   - Modify `LLM_CONFIGS` in `benchmark.py` to add or remove models.
   - Adjust `TARGET_LANGUAGE` as needed.

3. **Run the Benchmark**:
   ```bash
   python benchmark.py
   ```
   Each test case will display the source text, the reference translation, and the output of each model alongside its similarity, BLEU, and ROUGE-L scores.

<br>

---

<br>

## **Configuration**

- **`TARGET_LANGUAGE`** in `benchmark.py` dictates the translation target language.
- **`LLM_CONFIGS`** in `benchmark.py` define the providers and models tested.
- API keys are retrieved via environment variables to maintain security and flexibility.
