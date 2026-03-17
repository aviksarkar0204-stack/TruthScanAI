from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

def to_lower(text):
    return text.lower()

def remove_html(text):
    return re.sub('<.*?>', ' ', text)

def remove_punctuation(text):
    return re.sub('[^a-zA-Z]', ' ', text)

def tokenization(text):
    return word_tokenize(text)

def remove_stopwords(text):
    stop_words = stopwords.words('english')
    return [word for word in text if word not in stop_words]

def steming(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text]

def list_to_string(text):
    return ' '.join(text)

def preprocessing(text):
    text = remove_html(text)
    text = to_lower(text)
    text = remove_punctuation(text)
    text = tokenization(text)
    text = remove_stopwords(text)
    text = steming(text)
    return list_to_string(text)