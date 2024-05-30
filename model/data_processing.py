import os
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from scipy.sparse import hstack
from sklearn.naive_bayes import MultinomialNB
from model.preprocessing import preprocess_text

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
csv_files = [f for f in os.listdir(DATA_DIR_UNPROCESSED) if f.endswith('.csv')]


def encode_label(label):
    if type(label) is int:
        return label
    else:
        if label.lower() in ['phishing email', 'phishing', '1']:
            return 1
        elif label.lower() in ['safe email', 'safe', '0']:
            return 0
        else:
            raise ValueError(f"Unknown label: {label}")


def standardize_and_encode(df, column_mapping):
    df = df.rename(columns=column_mapping)
    df['label'] = df['label'].apply(encode_label)
    df = df[['sender', 'text_combined', 'label']]
    return df


dataframes = []
for file in csv_files:
    df = pd.read_csv(os.path.join(DATA_DIR_UNPROCESSED, file))
    if 'sender' not in df.columns:
        df['sender'] = ''
    df = standardize_and_encode(df, column_mapping)
    dataframes.append(df)

data = pd.concat(dataframes, ignore_index=True)
data['text_combined'] = data['text_combined'].fillna('')
data['sender'] = data['sender'].fillna('')

data['processed_text'] = data['text_combined'].apply(preprocess_text)
processed_csv_path = os.path.join(DATA_DIR_PROCESSED, 'processed_data.csv')
data.to_csv(processed_csv_path, index=False)

DATA_DIR_PROCESSED = os.path.join(os.path.dirname(__file__), 'data', 'preprocessed')

data = pd.read_csv(os.path.join(DATA_DIR_PROCESSED, 'processed_data.csv'), dtype={
            'processed_text': str,
            'text_content': str,
            'sender': str,
            'label': int
        },
        low_memory=False
        )

data['processed_text'] = data['processed_text'].fillna('')
data['sender'] = data['sender'].fillna('')

vectorizer = TfidfVectorizer()
sender_vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(data['processed_text'])
X_sender = sender_vectorizer.fit_transform(data['sender'])

X = hstack([X_text, X_sender])

joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
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
