require 'numo/narray'
require_relative '../common/time_layers'

class SimpleRnnlm
  def initialize(vocab_size, wordvec_size, hidden_size)
    v, d, h = vocab_size, wordvec_size, hidden_size
    ndn = Numo::DFloat.new
    sqrt = Numo::DFloat::Math.sqrt
    zeros = Numo::DFloat.zeros

    embed_w = ndn(v, d).rand_norm / 100
    rnn_wx = ndn(d, h).rand_norm / sqrt(d)
    rnn_wh = ndn(h, h) / sqrt(h)
    rnn_b = zeros(h)
    affine_w = ndn(h, v) / sqrt(h)
    affine_b = zeros(v)

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
