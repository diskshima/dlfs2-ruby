require 'numo/narray'

URL_BASE = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/'

KEY_FILE = {
  train: 'ptb.train.txt',
  test: 'ptb.test.txt',
  valid: 'ptb.valid.txt',
}

SAVE_FILE = {
  train: 'ptb.train.bin',
  test: 'ptb.test.bin',
  valid: 'ptb.valid.bin',
}

VOCAB_FILE = 'ptb.vocab.bin'

DATASET_DIR = File.dirname(File.absolute_path(__FILE__))

def download(file_name)
  file_path = File.join(DATASET_DIR, file_name)
  return if File.exist?(file_path)

  puts("Downloading #{file_name} ...")
  require 'open-uri'
  open(file_path, 'wb') do |f|
    f << open(URL_BASE + file_name).read
  end

  puts('Done')
end

def load_vocab
  vocab_path = File.join(DATASET_DIR, VOCAB_FILE)

  if File.exist?(VOCAB_FILE)
    File.open(vocab_path, 'rb') do |f|
      word_to_id, id_to_word = Marshal.load(f)
    end

    return [word_to_id, id_to_word]
  end

  word_to_id = {}
  id_to_word = {}
  data_type = :train
  file_name = KEY_FILE[data_type]
  file_path = File.join(DATASET_DIR, file_name)

  download(file_name)

  words = File.read(file_path).gsub("\n", '<eos>').strip.split

  words.each_with_index do |word, i|
    unless word_to_id.include?(word)
      tmp_id = word_to_id.length
      word_to_id[word] = tmp_id
      id_to_word[tmp_id] = word
    end
  end

  result = [word_to_id, id_to_word]
  File.open(vocab_path, 'wb') do |f|
    Marshal.dump(result, f)
  end

  result
end

# Load corpus, word-ID mappings
#
# @param data_type [Symbol] Data type to load. One of :train, :test or :valid/:val.
# @return [Array<Numo::NArray, Hash<String, Integer>, Hash<Integer, String>>] Corpus (text converted to IDs), word to ID, ID to word.
def load_data(data_type=:train)
  data_type = :valid if data_type == :val

  save_path = File.join(DATASET_DIR, SAVE_FILE[data_type])

  word_to_id, id_to_word = load_vocab

  if File.exist?(save_path)
    File.open(save_path, 'rb') do |f|
      corpus = Marshal.load(f)
      return [corpus, word_to_id, id_to_word]
    end
  end

  file_name = KEY_FILE[data_type]
  file_path = File.join(DATASET_DIR, file_name)
  download(file_name)

  words = File.read(file_path).gsub("\n", '<eos>').strip.split
  word_ids = words.map { |w| word_to_id[w] }
  corpus = Numo::UInt32.zeros(word_ids.length)
                       .append(word_ids)

  File.open(save_path, 'wb') do |f|
    Marshal.dump(corpus, f)
  end

  [corpus, word_to_id, id_to_word]
end

if __FILE__ == $0
  KEY_FILE.keys.each do |data_type|
    load_data(data_type)
  end
end