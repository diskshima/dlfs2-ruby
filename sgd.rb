class SGD
  def initialize(lr=0.01)
    @lr = lr
  end

  def update(params, grads)
    for i in (0...params.length) do
      params[i].inplace - @lr * grads[i]
    end
  end
end
