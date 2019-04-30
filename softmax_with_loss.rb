def softmax(x)
  ndim = x.ndim

  if ndim == 2
    x2 = x - x.max(axis: 1).expand_dims(1)
    exp = Numo::DFloat::Math.exp(x2)
    return x /= exp.sum(axis: 1).expand_dims(1)
  elsif ndim == 1
    x2 = x - x.max
    exp = Numo::DFloat::Math.exp(x2)
    return exp / exp.sum
  end
end

def cross_entropy_error(y, t)
  if y.ndim == 1
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
  end

  if t.size == y.size
    t = t.max_index(axis: 1)
  end

  batch_size = y.shape[0]

  -Numo::DFloat::Math.log(y[true, t] + 1e-7).sum / batch_size
end

class SoftmaxWithLoss
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
      @t = @t.max_index(axis: 1)
    end

    loss = cross_entropy_error(@y, @t)
    loss
  end

  def backward(dout=1)
    batch_size = @t.shape[0]

    dx = @y.copy
    dx[true, @t] -= 1
    dx *= dout
    dx /= batch_size
    dx
  end
end
