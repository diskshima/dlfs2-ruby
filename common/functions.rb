require 'numo/narray'

def softmax(x)
  ndim = x.ndim

  if ndim == 2
    x2 = x - x.max(axis: 1).expand_dims(1)
    exp = Numo::DFloat::Math.exp(x2)
    return exp / exp.sum(axis: 1).expand_dims(1)
  elsif ndim == 1
    x2 = x - x.max
    exp = Numo::DFloat::Math.exp(x2)
    return exp / exp.sum
  end
end

def argmax(x)
  x.max_index(axis: 1) % x.shape[1]
end

def cross_entropy_error(y, t)
  if y.ndim == 1
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
  end

  if t.size == y.size
    t = argmax(t)
  end

  batch_size = y.shape[0]

  idxs = to_full_index(t, y.shape[1])
  -Numo::DFloat::Math.log(y[idxs] + 1e-7).sum / batch_size
end

def to_full_index(t, cls_num)
  t.to_a.each_with_index.map do |val, index|
    cls_num * index + val
  end
end

def get_at_dim_index(x, dim_no, idx)
  ind = Array.new(x.ndim, true)
  ind[dim_no] = idx
  x[*ind]
end

# Choose `size` (default: 1) numbers of elements from `a` with the given
# probability (defaults to equal probability).
# `a` can either be an array or Integer in which case it will be treated as
# `(0...a).to_a`.
#
# @param a [Array or Integer] Array to choose from.
# @param size [Integer] Number of elements to pick. Default is 1.
# @param p [Array<Numeric>] Array of probabilities.
# @return [Array] Array of items chosen.
def random_choice(a, size: 1, p: nil)
  array = a.class == Integer ? (0...a).to_a : a

  if p
    raise 'The number of probabilities do not match the size of the array.' \
      if array.length != p.length
    val_to_weight = array.zip(p).to_h
    val_to_weight.max_by(size) { |_, weight| rand ** (1.0 / weight) }.map(&:first)
  else
    array.sample(size)
  end
end
