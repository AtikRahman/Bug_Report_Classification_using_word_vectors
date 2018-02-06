import pandas as pd
from gensim import models
import n_gram_selection



reviews = pd.read_excel('data/Mozilla_total.xlsx', sheet_name='Description')

x = reviews['description'].values
y = reviews['class'].values
length_of_corpus = len(x)
labels = list(sorted(set(y)))


text_of_labels = []
for label in labels:
    useful_words = n_gram_selection.useful_word_for_label(label, x, y)
    text_of_labels.append(useful_words)

print('useful words loaded...')

test_review = 'part code thing date string pseudo uid ic item s uniqu recent land nsiuuidgener bug s exist thunderbird branch build recommend rfc gener message id document jwz wrote point'
test_review = test_review.lower()
test_review = n_gram_selection.clean_str(test_review)
# test_review = stemming(test_review)

label_distance = {}
word2vec_model = models.KeyedVectors.load_word2vec_format('data/word_vectors.txt', binary=False)
print('word vectors loaded...')
all_similarity = []
for i in range(0, len(labels)):
    most_similar_word = ''
    most_similar_distance = 0.0
    similarity = 0.0
    for base_word in text_of_labels[i]:
        if base_word not in word2vec_model.wv.vocab:
            continue
        for word in test_review:
            if word not in word2vec_model.wv.vocab:
                continue
            similarity += word2vec_model.wv.similarity(word, base_word)
            # if similarity > most_similar_distance:
            #     most_similar_distance = similarity
            #     most_similar_word = word
    all_similarity.append(similarity)
    print('the similarity for label %d: %f' %(i+1, similarity))
print('ok')