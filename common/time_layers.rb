# frozen_string_literal: true

require 'numo/narray'
require_relative 'functions'
require_relative 'layers'

class RNN
  attr_accessor :params, :grads

  def initialize(wx, wh, b)
    @params = [wx, wh, b]
    @grads = [Numo::SFloat.zeros(wx.shape), Numo::SFloat.zeros(wh.shape),
              Numo::SFloat.zeros(b.shape)]
    @cache = nil
  end

  def forward(x, h_prev)
    wx, wh, b = @params
    t = h_prev.dot(wh) + x.dot(wx) + b
    h_next = Numo::SFloat::Math.tanh(t)

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
    @grads = [Numo::SFloat.zeros(wx.shape), Numo::SFloat.zeros(wh.shape),
              Numo::SFloat.zeros(b.shape)]
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
    hs = Numo::SFloat.zeros(n, t, h)

    if !@stateful || !@h
      @h = Numo::SFloat.zeros(n, h)
    end

    t.times do |ti|
      layer = RNN.new(*@params)
      @h = layer.forward(xs[true, ti, true], @h)
      hs[true, ti, true] = @h
      @layers.append(layer)
    end

    hs
  end

  def backward(dhs)
    wx, wh, b = @params
    n, t, h = dhs.shape
    d, h = wx.shape

    dxs = Numo::SFloat.zeros(n, t, d)
    dh = 0
    grads = [0, 0, 0]
    t.times.reverse_each do |ti|
      layer = @layers[ti]
      dx, dh = layer.backward(dhs[true, ti, true] + dh)

      dxs[true, ti, true] = dx

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

class LSTM
  attr_accessor :params, :grads

  def initialize(wx, wh, b)
    @params = [wx, wh, b]
    @grads = [Numo::SFloat.zeros(wx.shape), Numo::SFloat.zeros(wh.shape),
              Numo::SFloat.zeros(b.shape)]
    @cache = nil
  end

  def forward(x, h_prev, c_prev)
    wx, wh, b = @params
    n, h = h_prev.shape

    a = x.dot(wx) + h_prev.dot(wh) + b

    f = a[true, (0...h).to_a]
    g = a[true, (h...2*h).to_a]
    i = a[true, (2*h...3*h).to_a]
    o = a[true, (3*h...4*h).to_a]

    f = sigmoid(f)
    g = Numo::SFloat::Math.tanh(g)
    i = sigmoid(i)
    o = sigmoid(o)

    c_next = f * c_prev + g * i
    h_next = o * Numo::SFloat::Math.tanh(c_next)

    @cache = [x, h_prev, c_prev, i, f, g, o, c_next]
    [h_next, c_next]
  end

  def backward(dh_next, dc_next)
    wx, wh, b = @params
    x, h_prev, c_prev, i, f, g, o, c_next = @cache

    tanh_c_next = Numo::SFloat::Math.tanh(c_next)

    ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)

    dc_prev = ds * f

    di = ds * g
    df = ds * c_prev
    dou = dh_next * tanh_c_next
    dg = ds * i

    di.inplace * (i * (1 - i))
    df.inplace * (f * (1 - f))
    dou.inplace * (o * (1 - o))
    dg.inplace * (1 - g ** 2)

    da = Numo::NArray.hstack([df, dg, di, dou])

    dwh = h_prev.transpose.dot(da)
    dwx = x.transpose.dot(da)
    db = da.sum(axis: 0)

    @grads[0][] = dwx
    @grads[1][] = dwh
    @grads[2][] = db

    dx = da.dot(wx.transpose)
    dh_prev = da.dot(wh.transpose)

    [dx, dh_prev, dc_prev]
  end
end

class TimeLSTM
  attr_accessor :params, :grads
  def initialize(wx, wh, b, stateful = false)
    @params = [wx, wh, b]
    @grads = [Numo::SFloat.zeros(wx.shape), Numo::SFloat.zeros(wh.shape),
              Numo::SFloat.zeros(b.shape)]
    @layers = nil

    @h = nil
    @c = nil
    @dh = nil
    @stateful = stateful
  end

  def forward(xs)
    wx, wh, b = @params
    n, t, d = xs.shape
    h = wh.shape[0]

    @layers = []
    hs = Numo::SFloat.zeros(n, t, h)

    if !@stateful || !@h
      @h = Numo::SFloat.zeros(n, h)
    end

    if !@stateful || !@c
      @c = Numo::SFloat.zeros(n, h)
    end

    t.times do |ti|
      layer = LSTM.new(*@params)
      @h, @c = layer.forward(xs[true, ti, true], @h, @c)
      hs[true, ti, true] = @h

      @layers.append(layer)
    end

    hs
  end

  def backward(dhs)
    wx, wh, b = @params
    n, t, h = dhs.shape
    d = wx.shape[0]

    dxs = Numo::SFloat.zeros(n, t, d)
    dh = 0
    dc = 0

    grads = [0, 0, 0]

    t.times.reverse_each do |ti|
      layer = @layers[ti]
      dx, dh, dc = layer.backward(dhs[true, ti, true] + dh, dc)
      dxs[true, ti, true] = dx
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

  def set_state(h, c = nil)
    @h = h
    @c = c
  end

  def reset_state
    @h = nil
    @c = nil
  end
end

class TimeEmbedding
  attr_accessor :params, :grads

  def initialize(w)
    @params = [w]
    @grads = [Numo::SFloat.zeros(w.shape)]
    @layers = nil
    @w = w
  end

  def forward(xs)
    n, t = xs.shape
    v, d = @w.shape

    out = Numo::SFloat.zeros(n, t, d)
    @layers = []

    t.times do |ti|
      layer = Embedding.new(@w)
      out[true, ti, true] = layer.forward(xs[true, ti])
      @layers.append(layer)
    end

    out
  end

  def backward(dout)
    n, t, d = dout.shape

    grad = 0
    t.times do |ti|
      layer = @layers[ti]
      layer.backward(dout[true, ti, true])
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
    @grads = [Numo::SFloat.zeros(w.shape), Numo::SFloat.zeros(b.shape)]
    @x = nil
  end

  def forward(x)
    n, t, d = x.shape
    w, b = @params

    rest = x.size / (n * t)
    rx = x.reshape(n * t, rest)
    out = rx.dot(w) + b
    @x = x
    out_rest = out.size / (n * t)
    out.reshape(n, t, out_rest)
  end

  def backward(dout)
    x = @x
    n, t, d = x.shape
    w, b = @params

    dout_rest = dout.size / (n * t)
    dout = dout.reshape(n * t, dout_rest)
    rx = x.reshape(n * t, d)

    db = dout.sum(axis: 0)
    dw = rx.transpose.dot(dout)
    dx = dout.dot(w.transpose)
    dx = dx.reshape(*x.shape)

    @grads[0][] = dw
    @grads[1][] = db

    dx
  end
end

class TimeSoftmaxWithLoss
  attr_accessor :params, :grads

  def initialize
    @params = []
    @grads = []
    @cache = nil
    @ignore_label = -1
  end

  def forward(xs, ts)
    n, t, v = xs.shape

    if ts.ndim == 3
      ts = argmax(ts, axis: 2)
    end

    mask = Numo::UInt32.cast(ts.ne(@ignore_label))

    xs = xs.reshape(n * t, v)
    ts = ts.reshape(n * t)
    mask = mask.reshape(n * t)

    ys = softmax(xs)
    ls = Numo::SFloat::Math.log(paired_access(ys, Numo::UInt32.new(n * t).seq, ts))
    ls *= mask
    loss = -ls.sum
    loss /= mask.sum

    @cache = [ts, ys, mask, [n, t, v]]
    loss
  end

  def backward(dout = 1.0)
    ts, ys, mask, shapes = @cache
    n, t, v = shapes

    dx = ys
    full_idxs = paired_access_idxs(dx, Numo::UInt32.new(n * t).seq, ts)
    dx[full_idxs] -= 1
    dx *= dout
    dx /= mask.sum
    dx *= mask[false, :new]

    dx = dx.reshape(n, t, v)
    dx
  end
end

# Dropout
class TimeDropout
  attr_accessor :params, :grads, :train_flg

  def initialize(dropout_ratio = 0.5)
    @params = []
    @grads = []
    @dropout_ratio = dropout_ratio
    @mask = nil
    @train_flg = true
  end

  def forward(xs)
    if @train_flg
      flg = Numo::SFloat.cast(Numo::SFloat.new_like(xs).rand > @dropout_ratio)
      scale = 1.0 / (1.0 - @dropout_ratio)
      @mask = flg * scale
      xs * @mask
    else
      xs
    end
  end

  def backward(dout)
    dout * @mask
  end
end
