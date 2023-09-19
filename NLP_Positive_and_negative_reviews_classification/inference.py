import pandas as pd
import joblib
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from data_preprocessing import clean_text  


model_filename = 'random_forest_model.pkl'
model = joblib.load(model_filename)

if len(sys.argv) != 3:
    print("Usage: python3 inference.py input_file.csv output_file.csv")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

data = pd.read_csv(input_file)

data['text'] = data['text'].apply(clean_text)

tfid_filename = 'tfidf_vectorizer.pkl'
tfid = joblib.load(tfid_filename)

X_test = tfid.transform(data['text'])

predictions = model.predict(X_test)

predicted_sentiment = ["Negative" if label == 0 else "Positive" for label in predictions]


data['predicted_sentiment'] = predicted_sentiment

result_df = data[['id', 'predicted_sentiment']]

result_df.to_csv(output_file, index=False)
