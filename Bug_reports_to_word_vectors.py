import numpy as np
from gensim import models
from nltk.tokenize import word_tokenize
import re
import string
import pandas as pd


def clean_str(text):
    # text = re.sub("\d", "", text)           #remove decimal number

    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return text



review_dataset = 'data/Mozilla_total.xlsx'
word_vector_dataset = 'data/word_vectors.txt'
reviews = pd.read_excel(review_dataset, sheet_name='Description')
text = reviews['description'].values

text = [clean_str(i) for i in text]
text = [word_tokenize(i) for i in text]

model = models.Word2Vec(text, size=50, min_count=2, workers=4)
model.wv.save_word2vec_format(word_vector_dataset, binary=False)
print('word vectors created')