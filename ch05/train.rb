require_relative '../common/optimizer'
require_relative '../common/trainer'
require_relative '../dataset/ptb'
require_relative './simple_rnnlm'

batch_size = 10
wordvec_size = 100
hidden_size = 100
time_size = 5
lr = 0.1
max_epoch = 100

corpus, word_to_id, id_to_word = load_data(:train)
corpus_size = 1_000
corpus = corpus[0...corpus_size]
vocab_size = corpus.max + 1
xs = corpus[0...-1]
ts = corpus[1...]

model = SimpleRnnlm.new(vocab_size, wordvec_size, hidden_size)
optimizer = SGD.new(lr)
trainer = RnnlmTrainer.new(model, optimizer)

trainer.fit(xs, ts, max_epoch, batch_size, time_size)
trainer.plot
