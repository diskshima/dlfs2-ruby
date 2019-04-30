require 'numo/narray'

class Sigmoid
  attr_accessor :params

  def initialize
    @params = []
    @grads = []
    @out = nil
  end

  def forward(x)
    @out = 1 / (1 + Numo::DFloat::Math.exp(-x))
    @out
  end

  def backward(dout)
    dx = dout * (1.0 - @out) * @out
    dx
  end
end

if __FILE__ == $0
  sig = Sigmoid.new
  p sig.forward(1.0)

  arr = Numo::DFloat.new(2, 4).seq
  puts 'Forward:'
  p sig.forward(arr)

  dout = 3.0
  puts "Backward with #{dout}:"
  p sig.backward(dout)
end
