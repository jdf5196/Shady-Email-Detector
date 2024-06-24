import os
import re

import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from preprocessing import preprocess_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
    df = df[['text_combined', 'label']]
    return df


data = pd.read_csv(csv_file)

data = standardize_and_encode(data, column_mapping)

data['text_combined'] = data['text_combined'].fillna('')

data['processed_text'] = data['text_combined'].apply(preprocess_text)
processed_csv_path = os.path.join(DATA_DIR_PROCESSED, 'preprocessed_data.csv')
data.to_csv(processed_csv_path, index=False)

data = pd.read_csv(os.path.join(DATA_DIR_PROCESSED, 'preprocessed_data.csv'), 
    dtype={
        'processed_text': str,
        'text_combined': str,
        'label': int
    },
    low_memory=False
)

data['processed_text'] = data['processed_text'].fillna('')

vectorizer = TfidfVectorizer()
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

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')


def clean_text_for_wordcloud(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    filtered_tokens = [word for word in tokens if len(word) > 1]
    return ' '.join(filtered_tokens)

# Word Cloud
data['text_combined'] = data['text_combined'].astype(str)
text = ' '.join(data[data['label'] == 1]['processed_text'].apply(clean_text_for_wordcloud))
wordcloud = WordCloud(width=1920, height=1080, max_words=100, background_color='white').generate(text)
plt.imshow(wordcloud, interpolation='bicubic')
plt.axis('off')
plt.title('Word Cloud for Shady Emails')
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
#
# ROC Curve
fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
