# frozen_string_literal: true

require 'numo/narray'
require_relative './base_model'

class RnnLm < BaseModel
  attr_accessor :params, :grad

  def initialize(vocab_size = 10_000, wordvec_size = 100, hidden_size = 100)
    v = vocab_size
    d = wordvec_size
    h = hidden_size
    rn = ->(r, c) { Numo::DFloat.new(r, c).rand_norm }

    embed_w = rn.call(v, d) / 100
    lstm_wx = rn.call(d, 4 * h) / Numo::DFloat::Math.sqrt(d)
    lstm_wh = rn.call(h, 4 * h) / Numo::DFloat::Math.sqrt(h)
    lstm_b = Numo::DFloat.zeros(4 * h)
    affine_w = rn.call(h, v) / Numo::DFloat::Math.sqrt(h)
    affine_b = Numo::DFloat.zeros(v)

    @layers = [
      TimeEmbedding.new(embed_w),
      TimeLSTM.new(lstm_wx, lstm_wh, lstm_b, stateful: true),
      TimeAffine.new(affine_w, affine_b)
    ]
    @loss_layer = TimeSoftmaxWithLoss.new
    @lstm_layer = @layers[1]

    @params = []
    @grads = []
    @layers.each do |layer|
      @params.append(layer.params)
      @grads.append(layer.grads)
    end
  end

  def predict(xs)
    @layers.each do |layer|
      xs = layer.forward(xs)
    end

    xs
  end

  def forward(xs, ts)
    score = predict(xs)
    loss = @loss_layer.forward(score, ts)
    loss
  end

  def backward(dout = 1.0)
    dout = @loss_layer.backward(dout)
    @layers.reverse_each do |layer|
      dout = layer.backward(dout)
    end
    dout
  end

  def reset_state
    @lstm_layer.reset_state
  end
end