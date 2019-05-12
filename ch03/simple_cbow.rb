require 'numo/narray'
require_relative '../common/functions'
require_relative '../common/layers'

class SimpleCBOW
  attr_accessor :params, :grads, :word_vecs

  def initialize(vocab_size, hidden_size)
    v = vocab_size
    h = hidden_size

    w_in = 0.01 * Numo::DFloat.new(v, h).rand_norm
    w_out = 0.01 * Numo::DFloat.new(h, v).rand_norm

    @in_layer0 = MatMul.new(w_in)
    @in_layer1 = MatMul.new(w_in)
    @out_layer = MatMul.new(w_out)
    @loss_layer = SoftmaxWithLoss.new

    layers = [@in_layer0, @in_layer1, @out_layer]
    @params = []
    @grads = []

    layers.each do |layer|
      @params += layer.params
      @grads += layer.grads
    end

    @word_vecs = w_in
  end

  def forward(contexts, target)
    h0 = @in_layer0.forward(get_at_dim_index(contexts, 1, 0))
    h1 = @in_layer1.forward(get_at_dim_index(contexts, 1, 1))
    h = (h0 + h1) * 0.5
    score = @out_layer.forward(h)
    loss = @loss_layer.forward(score, target)
    loss
  end

  def backward(dout = 1.0)
    ds = @loss_layer.backward(dout)
    da = @out_layer.backward(ds)
    da *= 0.5
    @in_layer1.backward(da)
    @in_layer0.backward(da)
    nil
  end
end
