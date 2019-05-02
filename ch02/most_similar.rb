require_relative '../common/util'

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = word_to_id.length
c = create_co_matrix(corpus, vocab_size)

most_similar('hello', word_to_id, id_to_word, c, top: 3)
