# frozen_string_literal: true

require 'numo/narray'
require 'gnuplot'

N = 2
H = 3
T = 50

dh = Numo::SFloat.ones(N, H)

wh = Numo::SFloat.new(H, H).rand_norm
# wh = Numo::SFloat.new(H, H).rand_norm * 0.35

norm_list = []
T.times do
  dh = dh.dot(wh.transpose)
  norm = Math.sqrt((dh**2).sum) / N
  norm_list.append(norm)
end

p norm_list

Gnuplot.open do |gp|
  Gnuplot::Plot.new(gp) do |plot|
    plot.xlabel('time step')
    plot.ylabel('norm')
    plot.xtics('(1, 5, 10, 15, 20)')

    plot.data << Gnuplot::DataSet.new(
      [(0..norm_list.length).to_a, norm_list]
    ) do |ds|
      ds.with = 'lines'
      ds.linewidth = 3
    end
  end
end
