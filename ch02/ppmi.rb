require_relative '../common/util'
require_relative 'util'

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = word_to_id.length
c = create_co_matrix(corpus, vocab_size)
w = ppmi(c, verbose: true)

puts 'Covariance matrix'
print_matrix(c, id_to_word)

puts '-' * 80

puts 'PPMI'
print_matrix(w, id_to_word)
