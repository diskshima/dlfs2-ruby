require_relative '../common/layers'

class TwoLayerNet
  attr_accessor :params, :grads

  def initialize(input_size, hidden_size, output_size)
    w1 = 0.01 * Numo::SFloat.new(input_size, hidden_size).rand_norm
    b1 = Numo::SFloat.zeros(hidden_size)
    w2 = 0.01 * Numo::SFloat.new(hidden_size, output_size).rand_norm
    b2 = Numo::SFloat.zeros(output_size)

    @layers = [
      Affine.new(w1, b1),
      Sigmoid.new(),
      Affine.new(w2, b2),
    ]

    @loss_layer = SoftmaxWithLoss.new

    @params = []
    @grads = []
    @layers.each do |layer|
      @params += layer.params
      @grads += layer.grads
    end
  end

  def predict(x)
    @layers.reduce(x) { |x, layer| layer.forward(x) }
  end

  def forward(x, t)
    score = predict(x)
    loss = @loss_layer.forward(score, t)
    loss
  end

  def backward(dout = 1.0)
    dout = @loss_layer.backward(dout)
    @layers.reverse.reduce(dout) { |dout, layer| layer.backward(dout) }
  end
end

if __FILE__ == $0
  Numo::NArray.srand
  x = Numo::SFloat.new(10, 2).rand(42)
  p x
  model = TwoLayerNet.new(2, 4, 3)
  p model.predict(x)
  p model.forward(x, Numo::NArray[2])
  p model.backward
end
