require_relative '../common/functions'
require_relative '../common/optimizer'
require_relative '../common/trainer'
require_relative '../common/util'
require_relative 'simple_cbow'

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1_000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = word_to_id.length
contexts, target = create_contexts_target(corpus, window_size: 1)
contexts = convert_one_hot(contexts, vocab_size)
target = convert_one_hot(target, vocab_size)

model = SimpleCBOW.new(vocab_size, hidden_size)
optimizer = Adam.new
trainer = Trainer.new(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot

word_vecs = model.word_vecs

id_to_word.each do |word_id, word|
  printf("%s: %s\n", word, get_at_dim_index(word_vecs, 0, word_id).to_a)
end
