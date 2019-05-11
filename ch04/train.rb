require 'numo/narray'
require_relative '../common/functions'
require_relative '../common/optimizer'
require_relative '../common/trainer'
require_relative '../common/util'
require_relative '../dataset/ptb'
require_relative 'cbow'

window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

corpus, word_to_id, id_to_word = load_data(:train)
vocab_size = word_to_id.length

contexts, target = create_contexts_target(corpus, window_size: window_size)

model = CBOW.new(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam.new
trainer = Trainer.new(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot

word_vecs = model.word_vecs
params = {}
params['word_vecs'] = word_vecs
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word

params_file = 'cbow_params.bin'
File.open(params_file, 'wb') do |f|
  Marshal.dump(params, f)
end
