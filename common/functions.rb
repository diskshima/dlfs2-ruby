# frozen_string_literal: true

require 'numo/narray'

# Returns the sigmoid of `x`.
#
# @param x [Numo::NArray] Matrix.
# @return [Numo::NArray] Sigmoid.
def sigmoid(x)
  1.0 / (1.0 + Numo::DFloat::Math.exp(-x))
end

# Returns the softmax of the given matrix.
#
# @param x [Numo::NArray] Matrix.
# @return [Numo::NArray] Softmax.
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

# Returns the indices of the maximum values along an axis.
#
# @param x [Numo::NArray] Matrix.
# @param axis [Integer] axis.
# @return [Array<Integer>] Indices with the maximum value.
def argmax(x, axis: nil)
  x.max_index(axis: axis) % x.shape[axis]
end

def cross_entropy_error(y, t)
  if y.ndim == 1
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
  end

  if t.size == y.size
    t = argmax(t, axis: 1)
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
# @return [Numo::NArray(view)] View of the extracted matrix.
def get_at_dim_index(x, dim_no, idxs)
  ind = dim_full_indices(x, dim_no, idxs)
  x[*ind]
end

# Return the full indexes for the `idxs` in the `dim_no` dimention of the matrix `x`.
#
# @param x [Numo::NArray] Matrix.
# @param dim_no [Integer] Dimension number.
# @param idxs [Integer or Array<Integer>] index(es) in the `dim_no` dimension.
# @return [Array<Integer>] Array of indices.
def dim_full_indices(x, dim_no, idxs)
  ind = Array.new(x.ndim, true)
  ind[dim_no] = idxs
  ind
end

# Calls `paired_access_idxs` and returns the corresponding elements from `x`.
#
# @see paired_access_idxs
#
# @param x     [Numo::NArray] Matrix.
# @param idxs1 [Array-like] indexes in the 1st dimension.
# @param idxs2 [Array-like] indexes in the 2nd dimension.
# @return [Numo::NArray] Elements at the given indexes.
def paired_access(x, idxs1, idxs2)
  x[paired_access_idxs(x, idxs1, idxs2)]
end

# Return the full indexes when given two arrays where each element with the
# same index corresponding to the 1st and 2nd index in the matrix `x`.
# This is the same as numpy's notation of `x[[0, 1, 2], [2, 2, 5]]` meaning
# the elements at [0, 2], [1, 2], [2, 5].
#
# @param x     [Numo::NArray] Matrix.
# @param idxs1 [Array-like] indexes in the 1st dimension.
# @param idxs2 [Array-like] indexes in the 2nd dimension.
# @return [Array<Integer>] Array of the indices.
def paired_access_idxs(x, idxs1, idxs2)
  tmp_idxs2 = idxs2.to_a
  full_idxs = idxs1.to_a.map.with_index do |val, i|
    x.shape[1] * val + tmp_idxs2[i]
  end

  full_idxs
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
