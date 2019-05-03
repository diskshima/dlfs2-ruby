require_relative '../common/util'
require_relative '../dataset/ptb'

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = load_data(:train)
vocab_size = word_to_id.length

puts 'Counting co-occurence ...'
c = create_co_matrix(corpus, vocab_size, window_size: window_size)

puts 'Calculating PPMI ...'
w = ppmi(c, verbose: true)

puts 'Calculating SVD ...'
s, u, v = svd(w, wordvec_size)

word_vecs = u[true, 0...wordvec_size]
queries = %w(you year car toyota)
queries.each do |query|
  most_similar(query, word_to_id, id_to_word, word_vecs, top: 5)
end
