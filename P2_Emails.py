# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

# Load your email dataset or create one with labeled emails (spam = 1, not spam = 0)

# Assuming you have a dataset with 'emails' (email content) and 'labels' (0 for not spam, 1 for spam)

# Read the dataset or load your data
# df = pd.read_csv('your_dataset.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['emails'], df['labels'], test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer to convert text data to numerical features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# K-Nearest Neighbors (KNN) Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (K)
knn_classifier.fit(X_train_tfidf, y_train)
knn_predictions = knn_classifier.predict(X_test_tfidf)

# Support Vector Machine (SVM) Classifier
svm_classifier = SVC(kernel='linear', C=1.0)  # You can choose different kernels and adjust C
svm_classifier.fit(X_train_tfidf, y_train)
svm_predictions = svm_classifier.predict(X_test_tfidf)

# Performance analysis
def evaluate_model(predictions, y_true):
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions)
    recall = recall_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)
    roc_auc = roc_auc_score(y_true, predictions)
    return accuracy, precision, recall, f1, roc_auc

# Evaluate KNN
knn_accuracy, knn_precision, knn_recall, knn_f1, knn_roc_auc = evaluate_model(knn_predictions, y_test)
print("K-Nearest Neighbors (KNN) Performance:")
print(f"Accuracy: {knn_accuracy}")
print(f"Precision: {knn_precision}")
print(f"Recall: {knn_recall}")
print(f"F1 Score: {knn_f1}")
print(f"ROC AUC: {knn_roc_auc}")

# Evaluate SVM
svm_accuracy, svm_precision, svm_recall, svm_f1, svm_roc_auc = evaluate_model(svm_predictions, y_test)
print("\nSupport Vector Machine (SVM) Performance:")
print(f"Accuracy: {svm_accuracy}")
print(f"Precision: {svm_precision}")
print(f"Recall: {svm_recall}")
print(f"F1 Score: {svm_f1}")
print(f"ROC AUC: {svm_roc_auc}")

# Cross-validation for a more comprehensive analysis
knn_cv_scores = cross_val_score(knn_classifier, X_train_tfidf, y_train, cv=5, scoring='accuracy')
svm_cv_scores = cross_val_score(svm_classifier, X_train_tfidf, y_train, cv=5, scoring='accuracy')

print("\nKNN Cross-Validation Scores:", knn_cv_scores)
print("Mean KNN Cross-Validation Score:", np.mean(knn_cv_scores))

print("\nSVM Cross-Validation Scores:", svm_cv_scores)
print("Mean SVM Cross-Validation Score:", np.mean(svm_cv_scores))
