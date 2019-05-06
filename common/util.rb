require 'numo/narray'
require 'numo/linalg/linalg'
Numo::Linalg::Loader.load_openblas '/usr/local/opt/openblas/lib'

# Create corpus and word-ID hash.
#
# @param text [String] Text to process.
# @return [Array<Numo::NArray, Hash<String, Integer>, Hash<Integer, String>>] Corpus (text converted to IDs), word to ID, ID to word.
def preprocess(text)
  text = text.downcase
             .gsub('.', ' .')
  words = text.split(' ')

  word_to_id = {}
  id_to_word = {}

  words.each do |word|
    unless word_to_id.include?(word)
      new_id = word_to_id.length
      word_to_id[word] = new_id
      id_to_word[new_id] = word
    end
  end

  corpus = Numo::NArray[*words.map { |w| word_to_id[w] }]

  [corpus, word_to_id, id_to_word]
end

# Cosine similarity.
#
# @param x [Numo::NArray] Vector 1.
# @param y [Numo::NArray] Vector 2.
# @param eps [Float] Epsilon to prevent divide by zero errors.
# @return [Float] Cosine similarity.
def cos_similarity(x, y, eps: 1e-8)
  nx = x / (Numo::DFloat::Math.sqrt((x ** 2).sum) + eps)
  ny = y / (Numo::DFloat::Math.sqrt((y ** 2).sum) + eps)
  nx.dot(ny)
end

# Create a co-occurence matrix
#
# @param corpus [Array-like] Corpus (an array of word IDs).
# @param vocab_size [Integer] Size of vocabulary.
# @param window_size [Integer] Window size. Window size = 1 means it consider 1 word before and after the target word as its context.
# @return [Numo::UInt32] Co-occurence matrix.
def create_co_matrix(corpus, vocab_size, window_size: 1)
  corpus_size = corpus.length
  co_matrix = Numo::UInt32.zeros(vocab_size, vocab_size)

  corpus.each_with_index do |word_id, idx|
    (1..window_size).each do |i|
      left_idx = idx - i
      right_idx = idx + i

      if left_idx >= 0
        left_word_id = corpus[left_idx]
        co_matrix[word_id, left_word_id] += 1
      end

      if right_idx < corpus_size
        right_word_id = corpus[right_idx]
        co_matrix[word_id, right_word_id] += 1
      end
    end
  end

  co_matrix
end

# Create context from the passed in corpus.
#
# @param corpus [Array<Integer>] Corpus.
# @param window_size [Integer] Window size of the corpus.
# @return [Array<Numo::UInt32>] Array with the first element being an array of
#   contexts and the second element being an array of the corresponding targets.
def create_contexts_target(corpus, window_size: 1)
  target = corpus[window_size...-window_size]
  contexts = []

  (window_size...corpus.length-window_size).each do |idx|
    cs = []
    (-window_size..window_size).each do |t|
      next if t == 0
      cs.append(corpus[idx + t])
    end
    contexts.append(cs)
  end

  [Numo::UInt32[*contexts], Numo::UInt32[*target]]
end

# Calculate the Positive Pointwise Mutual Information (PPMI)
#
# @param c [Numo::UInt32] Co-occurence matrix.
# @param verbose [Boolean] Verbose output.
# @param eps [Float] Epsilon to prevent divide by zero errors.
# @return [Numo::DFloat] PPMI matrix
def ppmi(c, verbose: false, eps: 1e-8)
  ppmi = Numo::DFloat.zeros(c.shape)
  n = c.sum
  s = c.sum(axis: 1)
  total = c.size
  count = 0

  c.shape[0].times do |i|
    c.shape[1].times do |j|
      pmi = Math.log2(c[i,j] * n / (s[i] * s[j] + eps))
      ppmi[i, j] = [0.0, pmi].max

      if verbose
        count += 1
        if count % (total / 100) == 0
          printf("%.1f%% done\n", 100 * count / total)
        end
      end
    end
  end

  ppmi
end

# Print most similar words.
#
# @param query [String] Query word.
# @param word_to_id [Hash<String, Integer>] Word to ID.
# @param id_to_word [Hash<Integer, String>] ID to word.
# @param word_matrix [Numo::UInt32] Word matrix each inner array will represent the word vector for the index.
# @param top [Integer] Count of words to return (sorted by similarity).
def most_similar(query, word_to_id, id_to_word, word_matrix, top: 5)
  unless word_to_id.include?(query)
    printf("%s not found\n", query)
    return
  end

  puts '[Query] ' + query

  query_id = word_to_id[query]
  query_vec = word_matrix[query_id, true]

  # similarity = word_matrix.map { |vec| cos_similarity(vec, query_vec) }

  similarity = []

  word_matrix.shape[0].times do |idx|
    vec = word_matrix[idx, true]
    similarity[idx] = cos_similarity(vec, query_vec)
  end

  sorted_indexes = similarity.map.with_index.sort.map(&:last).reverse

  count = 0
  sorted_indexes.each do |idx|
    next if idx == query_id

    printf(" %s: %s\n", id_to_word[idx], similarity[idx])
    count += 1
    break if count == top
  end
end

# Runs a Truncated SVD on the input matrix.
# Taken from
#   https://yoshoku.hatenablog.com/entry/2019/01/06/193347
#
# @param a [Numo::NArray] Input matrix
# @param k [Integer] Number of vectors to calculate.
# @return [Array<Numo::DFloat, Numo::DFloat, Numo::DFloat>]
#   Sigma (singular values), left-singular vector, right-singular vector transposed.
def svd(a, k)
  n_rows, = a.shape

  b = a.dot(a.transpose)

  vals_range = (n_rows - k)...n_rows
  evals, evecs = Numo::Linalg.eigh(b, vals_range: vals_range)

  s = Numo::NMath.sqrt(evals.reverse.dup)
  u = evecs.reverse(1).dup
  vt = (1.0 / s).diag.dot(u.transpose).dot(a)

  [s, u, vt]
end

# Applies gradient clipping.
# This applies in-place.
#
# @param grads [Array<Numo::DFloat>] Array of gradients.
# @param max_norm [Float] Maximum gradient.
def clip_grads(grads, max_norm)
  total_norm = grads.reduce(0) do |total, grad|
    total + (grad ** 2).sum
  end

  total_norm = Numo::DFloat::Math.sqrt(total_norm)

  rate = max_norm / (total_norm + 1e-6)

  if rate < 1
    grads.each { |grad| grad *= rate }
  end
end
