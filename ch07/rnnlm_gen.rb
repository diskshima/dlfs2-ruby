# frozen_string_literal: true

require 'numo/narray'
require_relative '../common/functions'
require_relative '../ch06/rnnlm'
require_relative '../ch06/better_rnnlm'

# Generator class using Rnnlm.
class RnnlmGen < Rnnlm
  def generate(start_id, skip_ids = nil, sample_size = 100)
    word_ids = [start_id]

    x = start_id
    while word_ids.length < sample_size
      x = Numo::UInt32[x].reshape(1, 1)
      score = predict(x)
      p = softmax(score.flatten)

      sampled = random_choice(p.length, size: 1, p: p)
      if !skip_ids || !skip_ids.include?(sampled)
        x = sampled
        word_ids.append(x[0])
      end
    end

    word_ids
  end

  def get_state
    [@lstm_layer.h, @lstm_layer.c]
  end

  def set_state(state)
    @lstm_layer.set_state(*state)
  end
end

# Generator class using BetterRnnlm.
class BetterRnnlmGen < BetterRnnlm
  def generate(start_id, skip_ids = nil, sample_size = 100)
    word_ids = [start_id]

    x = start_id
    while word_ids.length < sample_size
      x = Numo::UInt32[x].reshape(1, 1)
      score = predict(x).flatten
      p = softmax(score).flatten

      sampled = random_choice(p.length, size: 1, p: p)
      if !skip_ids || !skip_ids.include?(sampled)
        x = sampled
        word_ids.append(x[0])
      end
    end

    word_ids
  end

  def get_state
    states = []
    @lstm_layers.each do |layer|
      states.append([layer.h, layer.c])
    end
    states
  end

  def set_state(states)
    @lstm_layers.zip(states).each do |layer, state|
      layer.set_state(*state)
    end
  end
end
