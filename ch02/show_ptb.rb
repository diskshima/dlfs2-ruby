require_relative '../dataset/ptb'

corpus, word_to_id, id_to_word = load_data(:train)

puts "Corpus size: #{corpus.length}"
puts "Corpus[:30]: #{corpus[0...30].to_a}"
puts
puts "id_to_word[0]: #{id_to_word[0]}"
puts "id_to_word[1]: #{id_to_word[1]}"
puts "id_to_word[2]: #{id_to_word[2]}"
puts
puts "word_to_id['car']: #{word_to_id['car']}"
puts "word_to_id['happy']: #{word_to_id['happy']}"
puts "word_to_id['lexus']: #{word_to_id['lexus']}"
