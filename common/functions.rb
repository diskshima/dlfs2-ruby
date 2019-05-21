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

# Converts `t`-th class notation to its index in a Numo::NArray.
#
# This was necessary as Numo::NArray does not support notations like `x[[0,1], t]`
# to extract t-th item in the 1st and 2nd row of a matrix
# `x[to_full_index(t, x.shape[0]]` will be equal to Python's numpy notation of
# `x[x.shape[0], t]`.
#
# @param t [Array<Integer>] Vector of denoting each index of the corresponding class.
# @param cls_count [Integer] Number of elements each row has.
# @return [Array<Integer>] Index into a Numo::NArray.
def to_full_index(t, cls_count)
  t.to_a.each_with_index.map do |val, index|
    cls_count * index + val
  end
end

# Return a view for the `idxs` in the `dim_no` dimention of the matrix `x`.
# Essentially a shortcut for writing `x[true, idxs, true, ...]` to retrieve
# the `idxs` index values in the 2nd dimension.
#
# @param x [Numo::NArray] Matrix.
# @param dim_no [Integer] Dimension number.
# @param idxs [Integer or Array<Integer>] index(es) in the `dim_no` dimension.
def get_at_dim_index(x, dim_no, idxs)
  ind = Array.new(x.ndim, true)
  ind[dim_no] = idxs
  x[*ind]
end

# Choose `size` (default: 1) numbers of elements from `a` with the given
# probability (defaults to equal probability).
# `a` can either be an array or Integer in which case it will be treated as
# `(0...a).to_a`.
#
# Implementation is based on the Weighted Random Sampling by Efraimidis and Spirakis
# (https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30162-4_478).
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
