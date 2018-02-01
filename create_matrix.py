import pandas as pd
import n_gram_selection
from gensim import models
import numpy as np

# reviews = pd.read_excel('data/Mozilla_total.xlsx', sheet_name='Description')
#
# x = reviews['description'].values
# y = reviews['class'].values
# length_of_corpus = len(x)
# labels = list(sorted(set(y)))
#
# text_of_labels = []
# for label in labels:
#     useful_words = n_gram_selection.useful_word_for_label(label, x, y)
#     text_of_labels.append(useful_words)
#
# word_list = []
# for i in range(0, len(labels)):
#     word_list += text_of_labels[i]
#
# data_frame = pd.DataFrame([], columns=word_list)
# writer = pd.ExcelWriter('data/feature_matrix.xlsx', engine='xlsxwriter')
# data_frame.to_excel(writer, sheet_name='Sheet1')
# writer.save()

test_review = 'part code thing date string pseudo uid ic item s uniqu recent land nsiuuidgener bug s exist thunderbird branch build recommend rfc gener message id document jwz wrote point'

test_review = test_review.lower()
test_review = n_gram_selection.clean_str(test_review)
# test_review = stemming(test_review)

file = pd.read_excel('data/feature_matrix.xlsx', sheet_name='Sheet1')
header = list(file)



label_distance = {}
word2vec_model = models.KeyedVectors.load_word2vec_format('data/word_vectors.txt', binary=False)
print('word vectors loaded...')

row = np.zeros(len(header))
for word in test_review:
    most_similar = 0.0
    similar_index = 0
    if word not in word2vec_model.wv.vocab:
        continue
    for index, column in enumerate(header):
        if column not in word2vec_model.wv.vocab:
            continue
        similarity = word2vec_model.wv.similarity(word, column)
        if similarity > most_similar:
            similar_index = index
            most_similar = similarity
    row[similar_index] = most_similar
row = np.array([row])
data_frame = pd.DataFrame(data=row, columns=header)
writer = pd.ExcelWriter('data/feature_matrix.xlsx', engine='xlsxwriter')
data_frame.to_excel(writer, sheet_name='Sheet1')
writer.save()



print('its done ...')