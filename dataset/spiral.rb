require 'numo/narray'

def load_data(seed=1984)
  Numo::NArray.srand(seed)
  n = 100
  dim = 2
  cls_num = 3

  x = Numo::SFloat.zeros(n * cls_num, dim)
  t = Numo::Int32.zeros(n * cls_num, cls_num)

  cls_num.times.each do |j|
    n.times.each do |i|
      rate = i.to_f / n
      radius = 1.0 * rate
      theta = j * 4.0 + 4.0 * rate + Numo::SFloat.new.rand_norm * 0.2

      ix = n * j + i
      x[ix, true] = [radius * Numo::SFloat::Math.sin(theta),
                     radius * Numo::SFloat::Math.cos(theta)]
      t[ix, j] = 1
    end
  end

  [x, t]
end

if __FILE__ == $0
  p load_data
end
