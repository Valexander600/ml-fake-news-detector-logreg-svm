# 📰 Fake News Detection with Machine Learning

This project builds a machine learning model to classify news articles as real or fake using the Fake and Real News Dataset from Kaggle.

## 📂 Dataset
- Source: [Kaggle - Fake and Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Two CSV files: `Fake.csv` and `True.csv`

## 🧠 ML Workflow
- Text cleaning: lowercasing, punctuation removal, stopword removal
- Feature extraction using TF-IDF
- Model training: Logistic Regression, Naive Bayes, SVM, Random Forest
- Performance evaluation: accuracy, classification report, confusion matrix

## 📈 Results
- Logistic Regression and SVM achieved >90% accuracy
- Confusion matrix and ROC curve visualizations included

## 🚀 Future Enhancements
- Deploying the model with Streamlit for real-time predictions
- Using transformer models (like BERT) for deeper understanding

## 📁 Files
- `FakeNewsDetector.ipynb`: Colab notebook with code, analysis, and results
