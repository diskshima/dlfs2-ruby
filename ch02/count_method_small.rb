require 'gnuplot'
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

Gnuplot.open do |gp|
  Gnuplot::Plot.new(gp) do |plot|
    word_to_id.each do |word, word_id|
      puts "#{word}: #{u[word_id, 0]},#{u[word_id, 1]}"
      plot.set(:label, "\"#{word}\" at #{u[word_id, 0]},#{u[word_id, 1]} center")
    end
    plot.data << Gnuplot::DataSet.new([u[true, 0].to_a, u[true, 1].to_a]) do |ds|
      ds.title = 'Word'
      ds.with = 'points pt 2'
    end
  end
end
