require 'numo/narray'

class Affine
  attr_accessor :params, :grads

  def initialize(w, b)
    @params = [w, b]
    @grads = [Numo::DFloat.zeros(w.shape), Numo::DFloat.zeros(b.shape)]
    @x = nil
  end

  def forward(x)
    w, b = @params
    out = x.dot(w) + b
    @x = x
    return out
  end

  def backward(dout)
    w, b = @params
    dx = dout.dot(w.transpose)
    dw = @x.transpose.dot(dout)
    db = dout.sum(0)

    @grads[0][] = dw
    @grads[1][] = db

    dx
  end
end

if __FILE__ == $0
  x = Numo::Int64.new(1, 4).seq
  w = Numo::Int64.new(4, 2).seq
  b = Numo::Int64.new(1, 2).seq
  layer = Affine.new(w, b)
  puts 'Forward:'
  p layer.forward(x)

  dout = Numo::DFloat[1.2, 0.8]
  puts "Backward with #{dout.inspect.split("\n").join('->')}:"
  p layer.backward(dout)
end
