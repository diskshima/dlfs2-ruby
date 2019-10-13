# frozen_string_literal: true

require_relative '../dataset/ptb'
require_relative './rnnlm_gen'

_corpus, word_to_id, id_to_word = load_data(:train)
# vocab_size = word_to_id.length
# corpus_size = corpus.length

model = RnnlmGen.new
model.load_params('ch06/Rnnlm.bin')

start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$']
skip_ids = skip_words.map { |w| word_to_id[w] }

word_ids = model.generate(start_id, skip_ids)
txt = word_ids.map { |i| id_to_word[i] }.join(' ')
txt = txt.gsub(' <eos>', ".\n")
puts txt
