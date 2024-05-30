import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')


def preprocess_text(text):
    if type(text) is str:
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
        return ' '.join(stemmed_tokens)
    else:
        return ""
