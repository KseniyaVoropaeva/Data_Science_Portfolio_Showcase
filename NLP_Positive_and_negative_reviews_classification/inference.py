import pandas as pd
import joblib
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from data_preprocessing import clean_text  

'''
# Завантажте натреновану модель
model_filename = 'random_forest_model.pkl'
model = joblib.load(model_filename)

# Перевірка, чи передано правильну кількість аргументів командного рядка
if len(sys.argv) != 3:
    print("Usage: python3 inference.py input_file.csv output_file.csv")
    sys.exit(1)

# Отримайте імена файлів з аргументів командного рядка
input_file = sys.argv[1]
output_file = sys.argv[2]

# Завантажте вхідний файл з відгуками
data = pd.read_csv(input_file)

# Очистіть текст відгуків (використовуйте функцію clean_text)
data['text'] = data['text'].apply(clean_text)

tfid_filename = 'tfidf_vectorizer.pkl'
tfid = joblib.load(tfid_filename)

# Тепер ви можете використовувати tfid для векторизації тексту
X_test = tfid.transform(data['text'])

predictions = model.predict(X_test)

# Замініть результати 0 і 1 на "Negative" і "Positive"
predicted_sentiment = ["Negative" if label == 0 else "Positive" for label in predictions]


# Додайте стовпець із результатами до даних
data['predicted_sentiment'] = predicted_sentiment

# Створіть новий DataFrame зі стовпцями "id" та "sentiment"
result_df = data[['id', 'predicted_sentiment']]

# Запишіть результати класифікації у вихідний файл
result_df.to_csv(output_file, index=False)
'''