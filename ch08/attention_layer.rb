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
