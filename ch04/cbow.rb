require 'numo/narray'
require_relative '../common/layers'
require_relative 'negative_sampling_layer'

class CBOW
  attr_accessor :params, :grads, :word_vecs

  def initialize(vocab_size, hidden_size, window_size, corpus)
    v = vocab_size
    h = hidden_size

    w_in = 0.01 * Numo::SFloat.new(v, h).rand_norm
    w_out = 0.01 * Numo::SFloat.new(v, h).rand_norm

    @in_layers = []
    (2 * window_size).times do |i|
      layer = Embedding.new(w_in)
      @in_layers.append(layer)
    end
    @ns_loss = NegativeSamplingLoss.new(w_out, corpus, power: 0.75,
                                        sample_size: 5)

    layers = @in_layers + [@ns_loss]
    @params = []
    @grads = []
    layers.each do |layer|
      @params += layer.params
      @grads += layer.grads
    end

    @word_vecs = w_in
  end

  def forward(contexts, target)
    h = 0
    @in_layers.each_with_index do |layer, i|
      h += layer.forward(contexts[true, i])
    end
    h *= 1.0 / @in_layers.length
    loss = @ns_loss.forward(h, target)
    loss
  end

  def backward(dout = 1.0)
    dout = @ns_loss.backward(dout)
    dout *= 1.0 / @in_layers.length
    @in_layers.each do |layer|
      layer.backward(dout)
    end
    nil
  end
end
