# frozen_string_literal: true

require 'numo/narray'
require_relative '../common/optimizer'
require_relative '../common/trainer'
require_relative '../dataset/ptb'
require_relative './better_rnnlm'

batch_size = 20
wordvec_size = 650
hidden_size = 650
time_size = 35
lr = 20.0
max_epoch = 40
max_grad = 0.25
dropout = 0.5

corpus, word_to_id, = load_data(:train)
corpus_val, = load_data(:val)
corpus_test, = load_data(:test)

vocab_size = word_to_id.length
xs = corpus[0...-1]
ts = corpus[1..]

model = BetterRnnlm.new(vocab_size, wordvec_size, hidden_size, dropout)
optimizer = SGD.new(lr)
trainer = RnnlmTrainer.new(model, optimizer)

best_ppl = Numo::SFloat::MAX
max_epoch.times do |_epoch|
  trainer.fit(xs, ts, 1, batch_size, time_size, max_grad)

  model.reset_state
  ppl = eval_perplexity(model, corpus_val)
  puts "valid perplexity: #{ppl}"

  if best_ppl > ppl
    best_ppl = ppl
    model.save_params
  else
    lr /= 4.0
    optimizer.lr = lr
  end

  model.reset_state
  puts '-' * 50
end

model.reset_state
ppl_test = eval_perplexity(model, corpus_test)
puts "test perplexity: #{ppl_test}"
