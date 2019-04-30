require 'numo/narray'

class MatMul
  attr_accessor :params, :grads

  def initialize(w)
    @params = [w]
    @grads = [Numo::DFloat.zeros(w.shape)]
    @x = nil
  end

  def forward(x)
    w, = @params
    out = x.dot(w)
    @x = x
    out
  end

  def backward(dout)
    w, = @params
    dx = dout.dot(w.transpose)
    dw = @x.transpose.dot(dout)
    @grads[0] = dw
    dx
  end
end
