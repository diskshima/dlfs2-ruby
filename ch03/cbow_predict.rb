require 'numo/narray'
require_relative '../common/layers'

c0 = Numo::UInt32[1, 0, 0, 0, 0, 0, 0]
c1 = Numo::UInt32[0, 0, 1, 0, 0, 0, 0]

w_in = Numo::SFloat.new(7, 3).rand_norm
w_out = Numo::SFloat.new(3, 7).rand_norm

in_layer0 = MatMul.new(w_in)
in_layer1 = MatMul.new(w_in)
out_layer = MatMul.new(w_out)

h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1)
s = out_layer.forward(h)
p s
