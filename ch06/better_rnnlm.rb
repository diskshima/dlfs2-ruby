# frozen_string_literal: true

require 'numo/narray'
require_relative '../common/base_model'
require_relative '../common/time_layers'

# Rnnlm with two layers of LSTM, dropout and weight typing.
class BetterRnnlm < BaseModel
  attr_accessor :params, :grads

  def initialize(vocab_size = 10_000, wordvec_size = 650, hidden_size = 650,
                 dropout_ratio = 0.5)
    v, d, h = vocab_size, wordvec_size, hidden_size
    rn = ->(r, c) { Numo::DFloat.new(r, c).rand_norm }

    embed_w = rn.call(v, d) / 100
    lstm_wx1 = rn.call(d, 4 * h) / Numo::DFloat::Math.sqrt(d)
    lstm_wh1 = rn.call(h, 4 * h) / Numo::DFloat::Math.sqrt(h)
    lstm_b1 = Numo::DFloat.zeros(4 * h)
    lstm_wx2 = rn.call(h, 4 * h) / Numo::DFloat::Math.sqrt(d)
    lstm_wh2 = rn.call(h, 4 * h) / Numo::DFloat::Math.sqrt(h)
    lstm_b2 = Numo::DFloat.zeros(4 * h)
    affine_b = Numo::DFloat.zeros(v)

    @layers = [
      TimeEmbedding.new(embed_w),
      TimeDropout.new(dropout_ratio),
      TimeLSTM.new(lstm_wx1, lstm_wh1, lstm_b1, stateful: true),
      TimeDropout.new(dropout_ratio),
      TimeLSTM.new(lstm_wx2, lstm_wh2, lstm_b2, stateful: true),
      TimeDropout.new(dropout_ratio),
      TimeAffine.new(embed_w.transpose, affine_b)
    ]
    @loss_layer = TimeSoftmaxWithLoss.new
    @lstm_layers = [@layers[2], @layers[4]]
    @drop_layers = [@layers[1], @layers[3], @layers[5]]

    @params = []
    @grads = []

    @layers.each do |layer|
      @params += layer.params
      @grads += layer.grads
    end
  end

  def predict(xs, train_flg = false)
    @drop_layers.each do |layer|
      layer.train_flg = train_flg
    end

    @layers.each do |layer|
      xs = layer.forward(xs)
    end

    xs
  end

  def forward(xs, ts, train_flg = true)
    score = predict(xs, train_flg)
    loss = @loss_layer.forward(score, ts)
    loss
  end

  def backward(dout = 1.0)
    dout = @loss_layer.backward(dout)
    @layers.reverse.each do |layer|
      dout = layer.backward(dout)
    end
    dout
  end

  def reset_state
    @lstm_layers.map(&:reset_state)
  end
end
