require 'numo/narray'
require_relative 'functions'

class MatMul
  attr_accessor :params, :grads

  def initialize(w)
    @params = [w]
    @grads = [Numo::SFloat.zeros(w.shape)]
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
    @grads[0][] = dw
    dx
  end
end

class Affine
  attr_accessor :params, :grads

  def initialize(w, b)
    @params = [w, b]
    @grads = [Numo::SFloat.zeros(w.shape), Numo::SFloat.zeros(b.shape)]
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

class Sigmoid
  attr_accessor :params, :grads

  def initialize
    @params = []
    @grads = []
    @out = nil
  end

  def forward(x)
    @out = 1.0 / (1 + Numo::SFloat::Math.exp(-x))
    @out
  end

  def backward(dout)
    dx = dout * (1.0 - @out) * @out
    dx
  end
end

class SigmoidWithLoss
  attr_accessor :params, :grads

  def initialize
    @params = []
    @grads = []
    @loss = nil
    @y = nil
    @t = nil
  end

  def forward(x, t)
    @t = t
    @y = 1.0 / (1 + Numo::SFloat::Math.exp(-x))

    @loss = cross_entropy_error(Numo::NArray.column_stack([1 - @y, @y]), @t)
    @loss
  end

  def backward(dout = 1.0)
    batch_size = @t.shape[0]

    dx = (@y - @t) * dout / batch_size
    dx
  end
end

# Softmax layer
class Softmax
  attr_accessor :params, :grads

  def initialize
    @params = []
    @grads = []
    @out = nil
  end

  def forward(x)
    @out = softmax(x)
    @out
  end

  def backward(dout)
    dx = @out * dout
    sumdx = dx.sum(axis: 1, keepdims: true)
    dx -= @out * sumdx
    dx
  end
end

# Softmax + cross entropy loss layer
class SoftmaxWithLoss
  attr_accessor :params, :grads

  def initialize
    @params = []
    @grads = []
    @y = nil
    @t = nil
  end

  def forward(x, t)
    @t = t
    @y = softmax(x)

    if @t.size == @y.size
      @t = argmax(@t, axis: 1)
    end

    loss = cross_entropy_error(@y, @t)
    loss
  end

  def backward(dout = 1.0)
    batch_size = @t.shape[0]

    dx = @y.copy
    idxs = to_full_index(@t, @y.shape[1])
    dx[idxs] -= 1
    dx *= dout
    dx /= batch_size
    dx
  end
end

class Embedding
  attr_accessor :params, :grads

  def initialize(w)
    @params = [w]
    @grads = [Numo::SFloat.zeros(w.shape)]
    @idx = nil
  end

  def forward(idx)
    w, = @params
    @idx = idx
    out = get_at_dim_index(w, 0, idx)
    out
  end

  def backward(dout)
    dw, = @grads
    dw[] = 0
    get_at_dim_index(dw, 0, @idx).inplace + dout
    nil
  end
end
