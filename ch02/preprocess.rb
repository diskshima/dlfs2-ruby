require_relative '../common/util'

text = 'You say goodbye and I say hello'
corpus, word_to_id, id_to_word = preprocess(text)
p text
p corpus
puts id_to_word
