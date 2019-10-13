# frozen_string_literal: true

require_relative './rnnlm_gen'
require_relative '../dataset/ptb'

_corpus, word_to_id, id_to_word = load_data(:train)
# vocab_size = word_to_id.length
# corpus_size = corpus.length

model = BetterRnnlmGen.new
model.load_params('./BetterRnnlm.bin')

start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$']
skip_ids = skip_words.map { |w| word_to_id[w] }

word_ids = model.generate(start_id, skip_ids)
txt = word_ids.map { |i| id_to_word[i] }.join(' ')
txt = txt.gsub(' <eos>', ".\n")

puts txt

model.reset_state

start_words = 'the meaning of life is'
start_ids = start_words.split(' ').map { |w| word_to_id[w] }

start_ids[0...-1].each do |x|
  x = Numo::UInt32[x].reshape(1, 1)
  model.predict(x)
end

word_ids = model.generate(start_ids[-1], skip_ids)
word_ids = start_ids[0...-1] + word_ids
txt = word_ids.map { |i| id_to_word[i] }.join(' ')
txt = txt.gsub(' <eos>', ".\n")
puts '-' * 50
puts txt
