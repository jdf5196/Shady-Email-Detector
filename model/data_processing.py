import os
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from scipy.sparse import hstack
from sklearn.naive_bayes import MultinomialNB
from preprocessing import preprocess_text

nltk.download('punkt')
nltk.download('stopwords')

column_mapping = {
    'text_combined': 'text_combined',
    'Email Text': 'text_combined',
    'body': 'text_combined',
    'Email Type': 'label',
    'label': 'label',
    'Label': 'label'
}

DATA_DIR_UNPROCESSED = os.path.join(os.path.dirname(__file__), 'data', 'unprocessed')
DATA_DIR_PROCESSED = os.path.join(os.path.dirname(__file__), 'data', 'preprocessed')
csv_file = os.path.join(DATA_DIR_UNPROCESSED, 'Phishing_Email.csv')

data = pd.read_csv(csv_file)
data['text_combined'] = data['text_combined'].fillna('')

data['processed_text'] = data['text_combined'].apply(preprocess_text)
processed_csv_path = os.path.join(DATA_DIR_PROCESSED, 'preprocessed_data.csv')
data.to_csv(processed_csv_path, index=False)

data = pd.read_csv(os.path.join(DATA_DIR_PROCESSED, 'preprocessed_data.csv'), 
    dtype={
        'processed_text': str,
        'text_content': str,
        'label': int
    },
    low_memory=False
)

data['processed_text'] = data['processed_text'].fillna('')

vectorizer = TfidfVectorizer()
sender_vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['processed_text'])

joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

model = MultinomialNB()
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1', verbose=1)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

joblib.dump(best_model, 'phishing_model.pkl')
joblib.dump(sender_vectorizer, 'tfidf_sender_email_vectorizer.pkl')

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
