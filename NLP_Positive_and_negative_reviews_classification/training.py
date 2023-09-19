import pandas as pd
from data_preprocessing import clean_text  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from memory_profiler import memory_usage
import timeit
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
import joblib



labels_file_path = 'labels.csv'
reviews_file_path = 'reviews.csv'

labels_df = pd.read_csv(labels_file_path, delimiter=',')
reviews_df = pd.read_csv(reviews_file_path, delimiter=',')

df = pd.merge(reviews_df, labels_df, on='id', how='inner')

df["text"] = df["text"].apply(clean_text)

rows_to_drop = [69, 131, 177]
df = df.drop(rows_to_drop)

tfid = TfidfVectorizer()
X = tfid.fit_transform(df['text'])

tfidf_filename = 'tfidf_vectorizer.pkl'
joblib.dump(tfid, tfidf_filename)


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['sentiment'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
}

rf_classifier = RandomForestClassifier()


memory_usage_before = memory_usage()
start_time = timeit.default_timer()

grid_search_rf = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='balanced_accuracy')
grid_search_rf.fit(X_train, y_train)

end_time = timeit.default_timer()
memory_usage_after = memory_usage()

print("Best Parameters for Random Forest:", grid_search_rf.best_params_)
print("Best Balanced Accuracy:", grid_search_rf.best_score_)
print("Memory Usage (MB):", max(memory_usage_after) - max(memory_usage_before))
print("Execution Time (s):", end_time - start_time)

y_pred = grid_search_rf.predict(X_val)

balanced_acc = balanced_accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted')
recall = recall_score(y_val, y_pred, average='weighted')
f1 = f1_score(y_val, y_pred, average='weighted')

print("Balanced Accuracy on Validation Data:", balanced_acc)
print("Precision on Validation Data:", precision)
print("Recall on Validation Data:", recall)
print("F1-Score on Validation Data:", f1)

model_filename = 'random_forest_model.pkl'
joblib.dump(grid_search_rf.best_estimator_, model_filename)
print("Model saved as", model_filename)
