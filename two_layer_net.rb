require_relative 'affine'
require_relative 'sigmoid'

class TwoLayerNet
  attr_accessor :params

  def initialize(input_size, hidden_size, output_size)
    w1 = Numo::DFloat.new(input_size, hidden_size).rand
    b1 = Numo::DFloat.new(hidden_size).rand
    w2 = Numo::DFloat.new(hidden_size, output_size).rand
    b2 = Numo::DFloat.new(output_size).rand

    @layers = [
      Affine.new(w1, b1),
      Sigmoid.new(),
      Affine.new(w2, b2),
    ]

    @params = @layers.reduce([]) { |arr, layer| arr + layer.params }
  end

  def predict(x)
    @layers.reduce(x) { |x, layer| layer.forward(x) }
  end
end

if __FILE__ == $0
  Numo::NArray.srand
  x = Numo::DFloat.new(10, 2).rand(42)
  model = TwoLayerNet.new(2, 4, 3)
  p model.predict(x)
end
