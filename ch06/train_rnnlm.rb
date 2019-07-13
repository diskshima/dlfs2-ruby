# frozen_string_literal: true

require_relative '../common/optimizer'
require_relative '../common/trainer'
require_relative '../dataset/ptb'
require_relative './rnnlm'

batch_size = 20
wordvec_size = 100
hidden_size = 100
time_size = 35
lr = 20.0
max_epoch = 4
max_grad = 0.25

corpus, word_to_id, = load_data(:train)
corpus_test, = load_data(:test)
vocab_size = word_to_id.length
xs = corpus[0...-1]
ts = corpus[1..]

model = Rnnlm.new(vocab_size, wordvec_size, hidden_size)
optimizer = SGD.new(lr)
trainer = RnnlmTrainer.new(model, optimizer)

trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad,
            eval_interval: 20)
trainer.plot('[0:500]')

model.reset_state
ppl_test = eval_perplexity(model, corpus_test)
puts "test perplexity: #{ppl_test}"

model.save_params
