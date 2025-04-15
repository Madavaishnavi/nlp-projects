# Yelp Review Sentiment Analysis with Apache Spark

This project implements a scalable pipeline to perform sentiment classification on Yelp reviews using distributed machine learning techniques in Apache Spark. It includes complete end-to-end preprocessing, model training (SVM, Logistic Regression), evaluation, and a real-time streaming prediction deployment.

---

## Project Overview

Customer reviews on platforms like Yelp provide valuable feedback, but analyzing millions of unstructured reviews is non-trivial. This project tackles this challenge by building a distributed machine learning pipeline using Spark to classify Yelp reviews as positive or negative.

---

## Key Insights & Findings

- **Data Cleaning**: Reduced ~12M raw reviews to ~7M valid entries by filtering out reviews with non-numeric or invalid star ratings.
- **Elite User Patterns**: Users marked as 'elite' often left longer reviews and higher average ratings.
- **Polarity in Ratings**: Majority of ratings skewed towards 5★ and 1★ — indicating polarized experiences.
- **Model Results**:
  - **SVM** outperformed with F1 score ≈ **0.81**
  - **Logistic Regression** F1 ≈ **0.76**
  - SVM provided better generalization with sparse features
- **Streaming Pipeline**: Real-time prediction pipeline using Spark Streaming, reading socket inputs and outputting live sentiment labels.

---

## Folder Structure
```code
Yelp-Review-Sentiment-Analysis/
├── Yelp_Data_PreProcessing.py           # Merges and preprocesses datasets
├── Scalable_Project_ML_Model_Analysis.py # Training SVM and LR models
├── Scalable_Project_SVM_Model_Deployment.py # Saving models to HDFS
├── Prediction_Pipe_Line.py              # Real-time sentiment prediction
├── SentimentAnalysis on yelp business reviews.docx # Documentation on sentiment analysis
├── FinalReport.docx                     # Final project report
└── README.md                            # This file
```

---

## 🔍 Workflow Summary

1. **Data Merging & Cleaning**:
   - Yelp Review + User + Business datasets loaded as Spark DataFrames.
   - Non-numeric `stars` entries cleaned.
   - Reviews filtered for rating balance and class relevance.

2. **Text Preprocessing**:
   - Custom `UDFs` for punctuation, stopword removal, case normalization
   - Tokenization + TF-IDF via Spark MLlib pipeline

3. **Model Training**:
   - Models Trained: `SVMWithSGD`, `LogisticRegressionWithLBFGS`
   - Evaluation: `MulticlassClassificationEvaluator` for F1-score
   - Model artifacts stored in HDFS

4. **Streaming Prediction**:
   - Socket server ingests live review text
   - Same preprocessing pipeline applied
   - Trained model outputs sentiment prediction (positive/negative)

---

## Technologies & Tools

- **Frameworks**: Apache Spark (MLlib, SQL, Streaming), HDFS
- **Languages**: Python (PySpark), Spark UDFs
- **Libraries**: NLTK, scikit-learn, NumPy, matplotlib
- **Streaming**: `Spark StreamingContext`, `socketTextStream`
- **Storage**: Hadoop HDFS for model checkpointing

---

## Models

| Model               | F1 Score |
|--------------------|----------|
| Logistic Regression| 0.76     |
| SVM                | 0.81     |

---

## Visuals & Insights

- Distribution of review lengths
- Rating heatmaps by business category
- Weekly and hourly trends of reviews (via RDD transformations)
- Choropleth maps of review activity by state

---

## Data Source

- **Yelp Open Dataset**  
  ([https://www.yelp.com/dataset](https://www.yelp.com/dataset))

---

## How to Run

1. Run `Yelp_Data_PreProcessing.py` to prepare cleaned DataFrame
2. Train models using `Scalable_Project_ML_Model_Analysis.py`
3. Save model artifacts via `Scalable_Project_SVM_Model_Deployment.py`
4. Run `Prediction_Pipe_Line.py` and start socket input:
   ```bash
   nc -lk 9999
