# frozen_string_literal: true

class Sequence
  def initialize
    @id_to_char = {}
    @char_to_id = {}
  end

  def update_vocab(txt)
    chars = txt.chars

    chars.each do |char|
      unless char_to_id.include?(char)
        tmp_id = char_to_id.size
        @char_to_id[char] = tmp_id
        @id_to_char[tmp_id] = char
      end
    end
  end

  def load_data(file_name = 'addition.txt', seed = 1984)
    file_path = Fie.join(File.dirname(File.absolute_path(__FILE__)), file_name)

    unless File.exist?(file_path)
      puts "No file: #{file_name}"
      return nil
    end

    questions = []
    answers = []

    text = File.open(file_path, 'r').read.gsub(/\r\n?/, "\n")
    text.each do |line|
      idx = line.index('_')
      questions.append(line[0...idx])
      answers.append(line[idx..])
    end

    questions.length.times do |i|
      q = quesitons[i]
      a = answers[i]
      _update_vocab(q)
      _update_vocab(a)
    end

    x = Numo::UInt32.zeros(questions.length, questions[0].length)
    t = Numo::UInt32.zeros(questions.length, answers[0].length)

    questions.each_with_index do |sentence, i|
      x[i] = sentence.map { |c| @char_to_id[c] }
    end

    answers.each_with_index do |sentence, i|
      t[i] = sentence.map { |c| @char_to_id[c] }
    end

    indices = (0...x.length).to_a

    indices = seed ? indices.shuffle(random: Random.new(seed)) : indices.shuffle

    x = get_at_dim_index(x, 0, indices)
    t = get_at_dim_index(t, 0, indices)

    split_at = x.length - x.length / 10
    x_train = x[0...split_at]
    x_test = x[split_at...-1]
    t_train = t[0...split_at]
    t_test = t[split_at...-1]

    [[x_train, t_train], [x_test, t_test]]
  end

  def get_vocab
    [@char_to_id, @id_to_char]
  end
end