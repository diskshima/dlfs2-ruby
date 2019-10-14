# frozen_string_literal: true

require 'numo/narray'
require 'gnuplot'
require 'numo/narray'
require_relative '../dataset/sequence'
require_relative '../common/optimizer'
require_relative '../common/trainer'
require_relative '../common/util'
require_relative '../ch07/seq2seq'
require_relative '../ch07/peeky_seq2seq'
require_relative './attention_seq2seq'

seq = Sequence.new
train_data, test_data = seq.load_data('date.txt')
x_train, t_train = train_data
x_test, t_test = test_data

char_to_id, id_to_char = seq.get_vocab

x_train = reverse_each_row(x_train)
x_test = reverse_each_row(x_test)

vocab_size = char_to_id.length
wordvec_size = 16
hidden_size = 256
batch_size = 128
max_epoch = 10
max_grad = 5.0

# model = Seq2seq.new(vocab_size, wordvec_size, hidden_size)
# model = PeekySeq2seq.new(vocab_size, wordvec_size, hidden_size)
model = AttentionSeq2seq.new(vocab_size, wordvec_size, hidden_size)

optimizer = Adam.new
trainer = Trainer.new(model, optimizer)

acc_list = []
max_epoch.times do
  trainer.fit(x_train, t_train, 1, batch_size, max_grad: max_grad)

  correct_num = 0

  x_test.shape[0].times do |i|
    question = x_test[i, true].expand_dims(0)
    correct = t_test[i, true].expand_dims(0)
    verbose = i < 10
    correct_num += eval_seq2seq(model, question, correct, id_to_char,
                                verbose: verbose, is_reverse: true)
  end

  acc = correct_num.to_f / x_test.shape[0]
  acc_list.append(acc)
  printf("val acc %.3f%\n", (acc * 100))
end

model.save_params

x = (0...acc_list.length).to_a
Gnuplot.open do |gp|
  Gnuplot::Plot.new(gp) do |plot|
    plot.xlabel('epochs')
    plot.ylabel('accuracy')
    plot.set(:yrange, '[0.05:1.05]')

    plot.data << Gnuplot::DataSet.new([x, acc_list]) do |ds|
      ds.with = 'points pt 2'
      ds.linewidth = 2
    end
  end
end
