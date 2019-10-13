# frozen_string_literal: true

require_relative '../dataset/sequence'

sequence = Sequence.new

trains, tests = sequence.load_data('addition.txt', seed: 1984)
x_train, t_train = trains
x_test, t_test = tests
_char_to_id, id_to_char = sequence.get_vocab

puts x_train.shape
puts t_train.shape
puts x_test.shape
puts t_test.shape

p x_train[0, true]
p t_train[0, true]

puts x_train[0, true].to_a.map { |c| id_to_char[c] }.join
puts t_train[0, true].to_a.map { |c| id_to_char[c] }.join
