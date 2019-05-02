require 'numo/gnuplot'
require 'numo/linalg/linalg'
Numo::Linalg::Loader.load_openblas '/usr/local/opt/openblas/lib'
require_relative '../common/util'
require_relative 'util'

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = word_to_id.length
c = create_co_matrix(corpus, vocab_size)
w = ppmi(c)

s, u, v = Numo::Linalg.svd(w)

puts('Covariance Matrix')
print_matrix(c, id_to_word)

puts('PPMI')
print_matrix(w, id_to_word)

puts('SVD')
print_matrix(u, id_to_word)

Numo::gnuplot do
  set(terminal: 'png')
  set(output: 'svd.png')
  word_to_id.each do |word, word_id|
    puts "#{word}: #{u[word_id, 0]},#{u[word_id, 1]}"
    set(label: word, at: "#{u[word_id, 0]},#{u[word_id, 1]} center")
  end
  plot([u[true, 0], u[true, 1], w: 'points', pt: 5])
end
