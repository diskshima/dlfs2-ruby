require_relative '../common/layers'

class SimpleSkipGram
  attr_accessor :params, :grads, :word_vecs

  def initialize(vocab_size, hidden_size)
    w_in = 0.01 * Numo::SFloat.new(vocab_size, hidden_size).rand_norm
    w_out = 0.01 * Numo::SFloat.new(hidden_size, vocab_size).rand_norm

    @in_layer = MatMul.new(w_in)
    @out_layer = MatMul.new(w_out)
    @loss_layer1 = SoftmaxWithLoss.new
    @loss_layer2 = SoftmaxWithLoss.new

    layers = [@in_layer, @out_layer]
    @params = []
    @grads = []

    layers.each do |layer|
      @params += layer.params
      @grads += layer.grads
    end

    @word_vecs = w_in
  end

  def forward(contexts, target)
    h = @in_layer.forward(target)
    s = @out_layer.forward(h)
    l1 = @loss_layer1.forward(s, get_at_dim_index(contexts, 1, 0))
    l2 = @loss_layer2.forward(s, get_at_dim_index(contexts, 1, 1))
    loss = l1 + l2
    loss
  end

  def backward(dout = 1.0)
    dl1 = @loss_layer1.backward(dout)
    dl2 = @loss_layer2.backward(dout)
    ds = dl1 + dl2
    dh = @out_layer.backward(ds)
    @in_layer.backward(dh)
    nil
  end
end
