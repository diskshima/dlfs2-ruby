# frozen_string_literal: true

require 'numo/narray'
require_relative '../common/layers'

# Class for a weighted sum.
class WeightSum
  attr_accessor :params, :grads

  def initialize
    @params = []
    @grads = []
    @cache = nil
  end

  def forward(hs, a)
    n, t, _h = hs.shape

    ar = a.reshape(n, t, 1)
    t = hs * ar
    c = t.sum(axis: 1)

    @cache = [hs, ar]
    c
  end

  def backward(dc)
    hs, ar = @cache
    n, t, h = hs.shape
    dt = dc.reshape(n, 1, h).repeat(t, axis: 1)
    dar = dt * hs
    dhs = dt * ar
    da = dar.sum(axis: 2)

    [dhs, da]
  end
end

# Attention Weight class.
class AttentionWeight
  attr_accessor :params, :grads

  def initialize
    @params = []
    @grads = []
    @softmax = Softmax.new
    @cache = nil
  end

  def forward(hs, h_in)
    n, _t, h = hs.shape

    hr = h_in.reshape(n, 1, h)
    t = hs * hr
    s = t.sum(axis: 2)
    a = @softmax.forward(s)

    @cache = [hs, hr]
    a
  end

  def backward(da)
    hs, hr = @cache
    n, t, h = hs.shape

    ds = @softmax.backward(da)
    dt = ds.reshape(n, t, 1).repeat(h, axis: 2)
    dhs = dt * hr
    dhr = dt * hs
    dh = dhr.sum(axis: 1)

    [dhs, dh]
  end
end

# Attention layer
class Attention
  attr_accessor :params, :grads

  def initialize
    @params = []
    @grads = []

    @attention_weight_layer = AttentionWeight.new
    @weight_sum_layer = WeightSum.new
    @attention_weight = nil
  end

  def forward(hs, h)
    a = @attention_weight_layer.forward(hs, h)
    out = @weight_sum_layer.forward(hs, a)
    @attention_weight = a
    out
  end

  def backward(dout)
    dhs0, da = @weight_sum_layer.backward(dout)
    dhs1, dh = @attention_weight_layer.backward(da)
    dhs = dhs0 + dhs1
    [dhs, dh]
  end
end

# Attention layer for multiple blocks.
class TimeAttention
  attr_accessor :params, :grads, :layers, :attention_weights

  def initialize
    @params = []
    @grads = []
    @layers = nil
    @attention_weights = nil
  end

  def forward(hs_enc, hs_dec)
    _n, t, _h = hs_dec.shape
    out = Numo::SFloat.zeros(hs_dec.shape)
    @layers = []
    @attention_weights = []

    t.times do |ti|
      layer = Attention.new
      out[true, ti, true] = layer.forward(hs_enc, hs_dec[true, ti, true])
      @layers.append(layer)
      @attention_weights.append(layer.attention_weight)
    end

    out
  end

  def backward(dout)
    _n, t, _h = dout.shape
    dhs_enc = 0
    dhs_dec = Numo::SFloat.zeros(dout.shape)

    t.times do |ti|
      layer = @layers[ti]
      dhs, dh = layer.backward(dout[true, ti, true])
      dhs_enc += dhs
      dhs_dec[true, ti, true] = dh
    end

    [dhs_enc, dhs_dec]
  end
end
