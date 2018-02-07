import n_gram_selection
from gensim import models

with open('data/pos.txt') as f:
    pos = f.readlines()

with open('data/neg.txt') as f:
    neg = f.readlines()

with open('data/english_stop_words.txt', 'r') as f:
    stop_words = [line.rstrip() for line in f]

pos = n_gram_selection.list_to_string(pos)
neg = n_gram_selection.list_to_string(neg)
common_words = n_gram_selection.common_word_pos_neg(pos, neg)
pos_useful_words = n_gram_selection.get_useful_word(pos, neg)
neg_useful_words = n_gram_selection.get_useful_word(neg, pos)
text_of_labels = []
text_of_labels.append(pos_useful_words)
text_of_labels.append(neg_useful_words)




test_review = 'this delicately observed story , deeply felt and masterfully stylized , is a triumph for its maverick director .'
test_review = test_review.lower()
test_review = n_gram_selection.clean_str(test_review)
test_review = n_gram_selection.remove_words(test_review, common_words)
# test_review = stemming(test_review)

label_distance = {}
word2vec_model = models.KeyedVectors.load_word2vec_format('data/glove.txt', binary=False)
print('word vectors loaded...')
all_similarity = []
for i in range(0, 2):
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