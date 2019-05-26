require 'numo/narray'
require_relative 'functions'
require_relative 'layers'

class RNN
  attr_accessor :params, :grads

  def initialize(wx, wh, b)
    @params = [wx, wh, b]
    @grads = [Numo::DFloat.zeros(wx.shape), Numo::DFloat.zeros(wh),
              Numo::DFloat.zeros(b.shape)]
    @cache = nil
  end

  def forward(x, h_prev)
    wx, wh, b = @params
    t = h_prev.dot(wh) + x.dot(wx) + b
    h_next = Numo::DFloat.tanh(t)

    @cache = [x, h_prev, h_next]
    h_next
  end

  def backward(dh_next)
    wx, wh, b = @params
    x, h_prev, h_next = @cache

    dt = dh_next * (1.0 - h_next ** 2)
    db = dt.sum(axis: 0)
    dwh = h_prev.transpose.dot(dt)
    dh_prev = dt.dot(wh.transpose)
    dwx = x.transpose.dot(dt)
    dx = dt.dot(wx.transpose)

    @grads[0][] = dwx
    @grads[1][] = dwh
    @grads[2][] = db

    [dx, dh_prev]
  end
end

class TimeRNN
  attr_accessor :params, :grads

  def initialize(wx, wh, b, stateful: false)
    @params = [wx, wh, b]
    @grads = [Numo::DFloat.zeros(wx.shape), Numo::DFloat.zeros(wh),
              Numo::DFloat.zeros(b.shape)]
    @layers = nil

    @h = nil
    @dh = nil
    @stateful = stateful
  end

  def forward(xs)
    wx, wh, b = @params
    n, t, d = xs.shape
    d, h = wx.shape

    @layers = []
    hs = Numo::DFloat.new(n, h)

    if !@stateful || !@h
      @h = Numo::DFloat.zeros(n, h)
    end

    T.times do |t|
      layer = RNN.new(*@params)
      @h = layer.forward(get_at_dim_index(x, 1, t), @h)
      get_at_dim_index(hs, 1, t).inplace = @h
      @layers.append(layer)
    end

    hs
  end

  def backward(dhs)
    wx, wh, b = @params
    n, t, h = dhs.shape
    d, h = wx.shape

    dxs = Numo::DFloat.new(n, t, d)
    dh = 0
    grads = [0, 0, 0]
    t.times.reverse_each do |t|
      layer = @layers[t]
      dx, dh = layer.backward(get_at_dim_index(dhs, 1, t) + dh)
      get_at_dim_index(dxs, 1, t).inplace = dx

      layer.grads.each_with_index do |grad, i|
        grads[i] += grad
      end
    end

    grads.each_with_index do |grad, i|
      @grads[i][] = grad
    end

    @dh = dh

    dxs
  end

  def set_state(h)
    @h = h
  end

  def reset_state
    @h = nil
  end
end

class TimeEmbedding
  attr_accessor :params, :grads

  def initialize(w)
    @params = [w]
    @grads = Numo::DFloat.zeros(w.shape)
    @layers = nil
    @w = w
  end

  def forward(xs)
    n, t = xs.shape
    v, d = @w.shape

    out = Numo::DFloat.new(n, t, d)
    @layers = []

    t.times do |t|
      layer = Embedding.new(@w)
      get_at_dim_index(out, 1, t).inplace = layer.forward(xs[true, t])
      @layers.append(layer)
    end

    out
  end

  def backward(dout)
    n, t, d = dout.shape

    grad = 0
    T.times do |t|
      layer = @layers[t]
      layer.backward(get_at_dim_index(dout, 1, t))
      grad += layer.grads[0]
    end

    @grads[0][] = grad
    nil
  end
end

class TimeAffine
  attr_accessor :params, :grads

  def initialize(w, b)
    @params = [w, b]
    @grads = [Numo::DFloat.zeros(w.shape), Numo::DFloat.zeros(b.shape)]
    @x = nil
  end

  def forward(x)
    n, t, d = x.shape
    w, b = @params

    rx = x.reshape(n * t, -1)
    out = rx.dot(w) + b
    @x = x
    out.reshape(n, t, -1)
  end

  def backward(dout)
    x = @x
    n, t, d = x.shape
    w, b = @params

    dout = dout.reshape(n * t, -1)
    rx = x.reshape(n * t, -1)

    db = dout.sum(axis: 0)
    dw = rx.transpose.dot(dout)
    dx = dout.dot(w.transpose)
    dx = dx.reshape(*x.shape)

    @grads[0][] = dw
    @grads[1][] = db

    dx
  end
end
