# frozen_string_literal: true

require 'numo/narray'
require 'numo/linalg/linalg'
Numo::Linalg::Loader.load_openblas '/usr/local/opt/openblas/lib'

# Create corpus and word-ID hash.
#
# @param text [String] Text to process.
# @return [Array<Numo::NArray, Hash<String, Integer>, Hash<Integer, String>>]
#         Corpus (text converted to IDs), word to ID, ID to word.
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
  nx = x / (Numo::SFloat::Math.sqrt((x**2).sum) + eps)
  ny = y / (Numo::SFloat::Math.sqrt((y**2).sum) + eps)
  nx.dot(ny)
end

# Convert corpus into one-hot encodings.
#
# @param corpus [Array-like] Corpus (an array of word IDs).
# @param vocab_size [Integer] Size of vocabulary.
# @return [Numo::UInt32] The corpus one-hot encoded (2 or 3 dimensions).
def convert_one_hot(corpus, vocab_size)
  n = corpus.shape[0]

  if corpus.ndim == 1
    one_hot = Numo::UInt32.zeros(n, vocab_size)
    corpus.each_with_index do |word_id, idx|
      one_hot[idx, word_id] = 1
    end
  elsif corpus.ndim == 2
    c = corpus.shape[1]
    one_hot = Numo::UInt32.zeros(n, c, vocab_size)

    n.times do |idx0|
      word_ids = corpus[idx0, true]
      word_ids.each_with_index do |word_id, idx1|
        one_hot[idx0, idx1, word_id] = 1
      end
    end
  end

  one_hot
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
# @param corpus [Array-like] Corpus (an array of word IDs).
# @param window_size [Integer] Window size of the corpus.
# @return [Array<Numo::UInt32>] Array with the first element being an array of
#   contexts and the second element being an array of the corresponding targets.
def create_contexts_target(corpus, window_size: 1)
  target = corpus[window_size...-window_size]
  contexts = []

  (window_size...(corpus.length - window_size)).each do |idx|
    cs = []
    (-window_size..window_size).each do |t|
      next if t.zero?
      cs.append(corpus[idx + t])
    end
    contexts.append(cs)
  end
  n_contexts = Numo::UInt32.zeros(contexts.length, contexts[0].length)
  n_contexts[] = contexts

  n_target = Numo::UInt32.zeros(target.length)
  n_target[] = target

  [n_contexts, n_target]
end

# Calculate the Positive Pointwise Mutual Information (PPMI)
#
# @param c [Numo::UInt32] Co-occurence matrix.
# @param verbose [Boolean] Verbose output.
# @param eps [Float] Epsilon to prevent divide by zero errors.
# @return [Numo::SFloat] PPMI matrix
def ppmi(c, verbose: false, eps: 1e-8)
  ppmi = Numo::SFloat.zeros(c.shape)
  n = c.sum
  s = c.sum(axis: 1)
  total = c.size
  count = 0

  c.shape[0].times do |i|
    c.shape[1].times do |j|
      pmi = Math.log2(c[i, j] * n / (s[i] * s[j] + eps))
      ppmi[i, j] = [0.0, pmi].max

      if verbose
        count += 1
        if (count % (total.to_f / 100)).zero?
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
# @param word_matrix [Numo::UInt32] Word matrix each inner array will represent
#        the word vector for the index.
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
# @return [Array<Numo::SFloat, Numo::SFloat, Numo::SFloat>]
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
# @param grads [Array<Numo::SFloat>] Array of gradients.
# @param max_norm [Float] Maximum gradient.
def clip_grads(grads, max_norm)
  total_norm = grads.reduce(0) do |total, grad|
    total + (grad**2).sum
  end

  total_norm = Numo::SFloat::Math.sqrt(total_norm)

  rate = max_norm / (total_norm + 1e-6)

  grads.each { |grad| grad.inplace * rate } if rate < 1
end

# Evalulate perplexity.
#
# @param [BaseModel] Model to evaluate.
# @param corpus Corpus.
# @param batch_size [Integer] Batch size.
# @param time_size [Integer]
def eval_perplexity(model, corpus, batch_size: 10, time_size: 35)
  puts 'evaluating perplexity ...'
  corpus_size = corpus.length
  total_loss = 0
  max_iters = (corpus_size - 1) / (batch_size * time_size)
  jump = (corpus_size - 1) / batch_size

  max_iters.times do |iters|
    xs = Numo::UInt32.zeros(batch_size, time_size)
    ts = Numo::UInt32.zeros(batch_size, time_size)
    time_offset = iters * time_size
    offsets = (0...batch_size).map { |i| time_offset + (i * jump) }

    time_size.times do |t|
      offsets.each_with_index do |offset, i|
        xs[i, t] = corpus[(offset + t) % corpus_size]
        ts[i, t] = corpus[(offset + t + 1) % corpus_size]
      end
    end

    loss = model.forward(xs, ts)
    total_loss += loss

    printf "\r%d / %d", iters, max_iters
  end

  puts ''
  ppl = Math.exp(total_loss / max_iters)
  ppl
end

# Print analogy i.e. "a to b" is "c to ?".
#
# @param a [String]
# @param b [String]
# @param c [String]
# @param word_to_id [Hash<String, Integer>] Word to ID.
# @param id_to_word [Hash<Integer, String>] ID to word.
# @param word_matrix [Numo::UInt32] Word matrix each inner array will represent
#        the word vector for the index.
# @param top [Integer] Number of top possibilities to show.
# @param answer [String] The word to compare how close the matrix is to the
#        query.
def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top: 5, answer: nil)
  all_found = true
  [a, b, c].each do |word|
    unless word_to_id.include?(word)
      puts("#{word} not found.")
      all_found = false
    end
  end

  return unless all_found

  puts("\n[analogy] #{a}:#{b} = #{c}:?")
  a_vec = word_matrix[word_to_id[a], true]
  b_vec = word_matrix[word_to_id[b], true]
  c_vec = word_matrix[word_to_id[c], true]
  query_vec = b_vec - a_vec + c_vec
  query_vec = normalize(query_vec)

  similarity = word_matrix.dot(query_vec).to_a

  if answer
    puts("===>#{answer}:#{word_matrix[word_to_id[answer]].dot(query_vec)}")
  end

  count = 0

  sorted_indexes = similarity.map.with_index.sort.map(&:last).reverse

  sorted_indexes.each do |i|
    # TODO: Deal with NaNs
    next if [a, b, c].include?(id_to_word[i])

    puts(" #{id_to_word[i]}: #{similarity[i]}")

    count += 1
    break if count >= top
  end
end

# Normalize the given matrix or vector.
#
# @param x [Numo::NArray] 1 or 2 dimention matrix.
# @return [Numo::NArray] The normalized 1 dimention result.
def normalize(x)
  if x.ndim == 2
    s = Numo::SFloat::Math.sqrt((x**2).sum(axis: 1))
    x / s.reshape(s.shape[0], 1)
  elsif x.ndim == 1
    s = Numo::SFloat::Math.sqrt((x**2).sum)
    x / s
  end
end

# Evaluate Sequence to sequence model.
#
# @param model [Seq2seq] Sequence to sequence model.
# @param question [Array<Integer>] Array of question character IDs.
# @param correct [Array<Integer>] Target value.
# @param id_to_char [Hash<Integer, String>>] ID to character hash.
# @param verbose [Boolean] Verbose mode. If true, will print result.
# @param is_reverse [Boolean] Reverse mode. If true, will treat the
#        questions as a reversed string.
# @return [Integer] 1 if guess is correct, 0 otherwise.
def eval_seq2seq(model, question, correct, id_to_char, verbose: false,
                 is_reverse: false)
  correct = correct.flatten
  start_id = correct[0]
  correct = correct[1..-1]
  guess = model.generate(question, start_id, correct.length)

  question = question.flatten.to_a.map { |c| id_to_char[c.to_i] }.join('')
  correct = correct.flatten.to_a.map { |c| id_to_char[c.to_i] }.join('')
  guess = guess.flatten.to_a.map { |c| id_to_char[c.to_i] }.join('')

  if verbose
    if is_reverse
      question = question.reverse
    end

    colors = { ok: "\033[92m", fail: "\033[91m", close: "\033[0m" }
    puts "Q: #{question}"
    puts "T: #{correct}"

    is_windows = Gem::Platform.local.os =~ /mswin/

    if correct == guess
      mark = colors[:ok] + '☑' + colors[:close]
      if is_windows
        mark = 'O'
      end
    else
      mark = colors[:fail] + '☒' + colors[:close]
      if is_windows
        mark = 'x'
      end
    end
    puts mark + ' ' + guess
    puts '---'
  end

  guess == correct ? 1 : 0
end

# Reverse each row of the 2-dimension matrix.
#
# @param x [Numo::SFloat] 2-dimension matrix.
# @return [Numo::SFloat] The 2-dimension matrix with each row reversed.
def reverse_each_row(x)
  rows = x.to_a.map(&:reverse)
  Numo::SFloat[*rows]
end
