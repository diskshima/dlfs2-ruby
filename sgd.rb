class SGD
  def initialize(lr=0.01)
    @lr = lr
  end

  def update(params, grads)
    for i in [0...len(params)] do
      params[i] -= @lr * grads[i]
    end
  end
end
