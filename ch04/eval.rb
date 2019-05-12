require_relative '../common/util'

SCRIPT_DIR = File.dirname(File.absolute_path(__FILE__))

bin_path = File.join(SCRIPT_DIR, 'cbow_params.bin')

params = nil
File.open(bin_path, 'rb') { |f| params = Marshal.load(f) }
word_vecs = params['word_vecs']
word_to_id = params['word_to_id']
id_to_word = params['id_to_word']

queries = %w(you year car toyota)
queries.each do |query|
  most_similar(query, word_to_id, id_to_word, word_vecs, top: 5)
end

puts('-' * 50)

analogy('king', 'man', 'queen', word_to_id, id_to_word, word_vecs)
analogy('take', 'took', 'go', word_to_id, id_to_word, word_vecs)
analogy('car', 'cars', 'child', word_to_id, id_to_word, word_vecs)
analogy('good', 'better', 'bad', word_to_id, id_to_word, word_vecs)
