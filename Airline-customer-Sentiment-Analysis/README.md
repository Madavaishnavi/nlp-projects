# Airline Customer Experience Analysis with Classical ML & LLaMA 2

This project explores customer sentiment analysis in the airline industry by combining traditional machine learning approaches with transformer-based large language models. It includes scraping real-world airline reviews, NLP preprocessing, classification using multiple models, and integrating Meta’s LLaMA 2 (13B) model via Hugging Face for qualitative enhancement.

---

## Objectives

- Scrape airline reviews from Skytrax and similar portals
- Clean and preprocess raw text data
- Train multiple classification models on TF-IDF features
- Use LLaMA-2 from Hugging Face to enhance review interpretation
- Evaluate sentiment trends and prediction performance

---

## 📁 Project Structure
```code
Airline-Customer-Experience-Prediction/
├── Webscraping_Airline_Reviews.ipynb      # Web scraping logic
├── Airline_customer_Experience_Analysis.ipynb # ML & LLM sentiment modeling                          
└── README.md                              # This file

```
---

## Key Insights

- Negative reviews are significantly longer than positive ones, often containing multiple grievances.
- Frequent complaint themes include: **delays**, **baggage issues**, and **crew behavior**.
- Classical ML performed well:
  - Logistic Regression: ~75% accuracy
  - Naive Bayes: ~78% accuracy
- **LLaMA 2 (13B)** was used to:
  - Generate contextual embeddings
  - Experiment with zero-shot classification prompts
  - Observe deeper semantic understanding of airline feedback

---

## Tools & Libraries

- **Web Scraping**: `BeautifulSoup`, `requests`
- **Text Processing**: `nltk`, `re`, `stopwords`, `tokenization`
- **ML Models**: `scikit-learn` – Logistic Regression, Naive Bayes, SVM
- **LLM Integration**:
  - `transformers`
  - `huggingface_hub`
  - `torch` with CUDA support
- **Visualization**: `seaborn`, `matplotlib`

---

## Hugging Face & LLaMA 2 Integration

```python
from huggingface_hub import notebook_login
notebook_login()

from torch import cuda
model_id = 'meta-llama/Llama-2-13b-chat-hf'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
```
- Used meta-llama/Llama-2-13b-chat-hf via Hugging Face Hub

- Conducted qualitative analysis using embeddings and zero-shot prompting

- CUDA support enabled efficient GPU-backed inference

## Model Comparison


| Model               | Accuracy | F1 Score |
|--------------------|----------|----------|
| Logistic Regression| 75%      | 0.73     |
| Naive Bayes        | 78%      | 0.76     |
| LLaMA 2 (13B)       | Qualitative | ✅  |


## Data Source
Reviews were scraped from airline feedback sites including:

- [Skytrax](https://www.airlinequality.com/)

- Additional scraped sources via BeautifulSoup

## How to Run

1. Run Webscraping_Airline_Reviews.ipynb to collect review data.

2. Process and classify reviews using Airline_customer_Experience_Analysis.ipynb.

3. To use LLaMA 2:

    - Login via Hugging Face

    - Ensure torch and CUDA are installed

    - Load meta-llama/Llama-2-13b-chat-hf model from Hugging Face


 A real-world application that blends traditional ML techniques with cutting-edge LLMs to analyze sentiment in airline customer experiences.
