require 'gnuplot'
require 'numo/narray'
require_relative '../common/optimizer'
require_relative '../dataset/ptb'
require_relative './simple_rnnlm'

batch_size = 10
wordvec_size = 100
hidden_size = 100
time_size = 5
lr = 0.1
max_epoch = 100

corpus, word_to_id, id_to_word = load_data('train')
corpus_size = 1_000
corpus = corpus[0...1_000]
vocab_size = max(corpus) + 1

xs = corpus[0...-1]
ts = corpus[1..-1]
data_size = xs.length
puts "corpus size: #{corpus_size}, vocabulary size: #{vocab_size}"

max_iters = data_size / (batch_size * time_size)
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

model = SimpleRnnlm.new(vocab_size, wordvec_size, hidden_size)
optimizer = SGD.new(lr)

jump = (corpus_size - 1) / batch_size
offsets = (0...batch_size).times { |i| i * jump }

max_epoch.times do |epoch|
  max_iters.times do |iter|
    batch_x = Numo::UInt32.new(batch_size, time_size).ones
    batch_t = Numo::UInt32.new(batch_size, time_size).ones

    time_size.times do |t|
      offsets.each_with_index do |offset, i|
        batch_x[i, t] = xs[(offset + time_idx) % data_size]
        batch_t[i, t] = ts[(offset + time_idx) % data_size]
      end
      time_idx += 1
    end

    loss = model.forward(batch_x, batch_t)
    model.backward
    optimizer.update(model.params, model.grads)
    total_loss += loss
    loss_count += 1
  end

  ppl = Numo::DFloat::Math.exp(total_loss / loss_count)
  printf("| epoch %d | perplexity %.2f\n", epoch + 1, ppl)
  ppl_list.append(ppl.to_f)
  total_loss = 0
  loss_count = 0
end

x = (0...ppl_list.length).to_a

Gnuplot.open do |gp|
  Gnuplot::Plot.new(gp) do |plot|
    plot.xlabel('epoch')
    plot.ylabel('perplexity')

    plot.data << Gnuplot::Dataset.new([x, ppl_list]) do |ds|
      ds.title = 'train'
    end
  end
end