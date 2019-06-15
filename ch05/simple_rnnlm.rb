require 'numo/narray'
require_relative '../common/time_layers'

class SimpleRnnlm
  attr_accessor :params, :grads

  def initialize(vocab_size, wordvec_size, hidden_size)
    v, d, h = vocab_size, wordvec_size, hidden_size
    rn = ->(r, c) { Numo::DFloat.new(r, c).rand_norm }

    embed_w = rn.call(v, d) / 100
    rnn_wx = rn.call(d, h)/ Numo::DFloat::Math.sqrt(d)
    rnn_wh = rn.call(h, h) / Numo::DFloat::Math.sqrt(h)
    rnn_b = Numo::DFloat.zeros(h)
    affine_w = rn.call(h, v) / Numo::DFloat::Math.sqrt(h)
    affine_b = Numo::DFloat.zeros(v)

    @layers = [
      TimeEmbedding.new(embed_w),
      TimeRNN.new(rnn_wx, rnn_wh, rnn_b, stateful: true),
      TimeAffine.new(affine_w, affine_b)
    ]
    @loss_layer = TimeSoftmaxWithLoss.new
    @rnn_layer = @layers[1]

    @params = []
    @grads = []
    @layers.each do |layer|
      @params += layer.params
      @grads += layer.grads
    end
  end

  def forward(xs, ts)
    @layers.each do |layer|
      xs = layer.forward(xs)
    end
    loss = @loss_layer.forward(xs, ts)
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
    @rnn_layer.reset_state
  end
end
