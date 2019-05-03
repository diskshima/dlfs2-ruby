require_relative '../common/util'
require_relative '../dataset/ptb'

# Load a marshalled file at 'path'. If the file does not exist, it will run the
# passed in block and save the returned value to 'path'.
#
# @param path [String] Path to file.
def load_or_run(path)
  result = nil
  if File.exist?(path)
    File.open(path, 'rb') { |f| result = Marshal.load(f) }
  else
    result = yield
    File.open(path, 'wb') { |f| Marshal.dump(result, f) }
  end
  result
end

SCRIPT_DIR = File.dirname(File.absolute_path(__FILE__))

window_size = 2
wordvec_size = 100
ppmi_file = 'ptb.ppmi.bin'
ppmi_path = File.join(SCRIPT_DIR, ppmi_file)

corpus, word_to_id, id_to_word = load_data(:train)
vocab_size = word_to_id.length

puts 'Counting co-occurence ...'
comatrix_path =
  File.join(SCRIPT_DIR, "ptb.comatrix-#{vocab_size}-#{window_size}.bin")
c = load_or_run(comatrix_path) do
  create_co_matrix(corpus, vocab_size, window_size: window_size)
end

puts 'Calculating PPMI ...'
w = load_or_run(ppmi_path) { ppmi(c, verbose: true) }

puts 'Calculating SVD ...'
sigma_path = File.join(SCRIPT_DIR, "ptb.sigma-#{wordvec_size}.bin")
u = load_or_run(sigma_path) do
  _, u2, _ = svd(w, wordvec_size)
  u2
end

word_vecs = u[true, 0...wordvec_size]
queries = %w(you year car toyota)
queries.each do |query|
  most_similar(query, word_to_id, id_to_word, word_vecs, top: 5)
end
