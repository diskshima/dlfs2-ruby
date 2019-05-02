require_relative '../common/util'

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = word_to_id.length
c = create_co_matrix(corpus, vocab_size)

max_word_len = word_to_id.keys.max_by { |w| w.length }.length

c.shape[0].times do |i|
  printf("%s: %s\n", id_to_word[i].ljust(max_word_len + 1), c[i, true].to_a)
end

c0 = c[word_to_id['you'], true]
c1 = c[word_to_id['i'], true]
printf("you vs i: #{cos_similarity(c0, c1)}\n")

c2 = c[word_to_id['hello'], true]
c3 = c[word_to_id['goodbye'], true]
printf("hello vs goodbye: #{cos_similarity(c2, c3)}\n")
