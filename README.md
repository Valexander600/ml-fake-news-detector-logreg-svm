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

## 📸 Screenshots

### 🔍 Distribution of Real vs Fake News
![Alt Text](FakevsRealNewsDistribution.png)

### 📊 Confusion Matrix
![Confusion Matrix](ROCCurve.png)

### 🧠 Model Accuracy Comparison
![Model Accuracy](FakevsRealDatagraph.png)

## 🚀 Future Enhancements
- Deploying the model with Streamlit for real-time predictions
- Using transformer models (like BERT) for deeper understanding
- ## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Valexander600/ml-fake-news-detector-logreg-svm.git
   cd ml-fake-news-detector-logreg-svm
jupyter notebook FakeNewsDetector.ipynb



## 📁 Files
- `FakeNewsDetector.ipynb`: Colab notebook with code, analysis, and results

- ## 📝 License

This project is licensed under the MIT License.

