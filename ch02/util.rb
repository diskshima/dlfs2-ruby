# Prints word matrix
#
# @param m [Numo::UInt32] Word matrix.
# @param id_to_word [Hash<Integer, String>] ID to word.
def print_matrix(m, id_to_word)
  max_word_len = id_to_word.values.max_by(&:length).length

  m.shape[0].times do |i|
    printf('%s: ', id_to_word[i].ljust(max_word_len + 1))
    m.shape[1].times do |j|
      print(', ') unless j == 0
      printf('%.4f', m[i, j])
    end
    puts ''
  end
end
