# frozen_string_literal: true

require 'gnuplot'
require 'numo/narray'
require_relative '../dataset/sequence'
require_relative '../common/optimizer'
require_relative '../common/trainer'
require_relative '../common/util'
require_relative './seq2seq'

def each_batch(x)
  batches = x.to_a.map do |batch|
    yield x[batch, true]
  end

  Numo::SFloat[*batches]
end

def reverse_batch(batch)
  row_count = batch.shape[0]

  (row_count - 1).downto(0).map do |row|
    batch[row]
  end
end

seq = Sequence.new
train_data, test_data = seq.load_data('addition.txt')
x_train, t_train = train_data
x_test, t_test = test_data

char_to_id, id_to_char = seq.get_vocab

is_reverse = true

if is_reverse
  x_train = each_batch(x_train) { |b| reverse_batch(b) }
  t_test = each_batch(x_test) { |b| reverse_batch(b) }
end

vocab_size = char_to_id.length
wordvec_size = 16
hidden_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0

model = Seq2seq.new(vocab_size, wordvec_size, hidden_size)
# model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)

optimizer = Adam.new
trainer = Trainer.new(model, optimizer)

acc_list = []
max_epoch.times do
  trainer.fit(x_train, t_train, 1, batch_size, max_grad: max_grad)

  correct_num = 0
  x_test.shape[0].times do |i|
    question = Numo::SFloat[x_test[i, true]]
    correct = Numo::SFloat[t_test[i, true]]
    verbose = i < 10
    correct_num += eval_seq2seq(model, question, correct, id_to_char,
                                verbose: verbose, is_reverse: is_reverse)
  end

  acc = correct_num.to_f / x_test.shape[0]
  acc_list.append(acc)
  printf("val acc %.3f%\n", (acc * 100))
end

x = (0...acc_list.length).to_a
Gnuplot.open do |gp|
  Gnuplot::Plot.new(gp) do |plot|
    plot.xlabel('epochs')
    plot.ylabel('accuracy')
    plot.set(:yrange, '[0:1.0]')

    plot.data << Gnuplot::DataSet.new([x, acc_list]) do |ds|
      ds.with = 'points pt 2'
      ds.linewidth = 2
    end
  end
end
