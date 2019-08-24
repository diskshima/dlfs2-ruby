# frozen_string_literal: true

# Implementation of Weighted Random Sampling.
class RandomSampling
  # Choose `size` (default: 1) numbers of elements from `a` with the given
  # probability.
  # `a` can either be an array or Integer in which case it will be treated as
  # `(0...a).to_a`.
  #
  # @param a [Array or Integer] Array to choose from.
  # @param size [Integer] Number of elements to pick. Default is 1.
  # @param p [Array<Numeric>] Array of probabilities.
  # @param replacement [Boolean] `true` if with replacement. Defaults to true.
  # @return [Array] Array of items chosen.
  def self.random_choice(a, size: 1, p:, replacement: true)
    if replacement
      random_choice_with_replacement(a, size: size, p: p)
    else
      random_choice_without_replacement(a, size: size, p: p)
    end
  end

  # Implementation is based on the Weighted Random Sampling by Efraimidis and
  # Spirakis (https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30162-4_478).
  def self.random_choice_with_replacement(a, size: 1, p: nil)
    array = a.class == Integer ? (0...a).to_a : a

    if p
      raise 'The number of probabilities do not match the size of the array.' \
        if array.length != p.length

      val_to_weight = array.zip(p).to_h
      val_to_weight.max_by(size) { |_, weight| rand**(1.0 / weight) }
                   .map(&:first)
    else
      array.sample(size)
    end
  end

  # Implementation is based on the Weighted Random Sampling from this SO
  # (https://stackoverflow.com/a/2149533).
  def self.random_choice_without_replacement(a, size: 1, p:)
    array = a.class == Integer ? (0...a).to_a : a
    items = array.zip(p)

    heap = rws_heap(items)

    size.times.map { rws_heap_pop(heap) }
  end

  Node = Struct.new(:w, :v, :tw)
  Rand = Random.new

  def self.rws_heap(items)
    h = [nil]
    items.each do |w, v|
      h.append(Node.new(w, v, w))
    end

    (h.length - 1).downto(2).each do |i|
      h[i >> 1].tw += h[i].tw
    end

    h
  end

  def self.rws_heap_pop(h)
    gas = h[1].tw * Rand.rand

    i = 1

    while gas >= h[i].w
      gas -= h[i].w
      i <<= 1
      if gas >= h[i].tw
        gas -= h[i].tw
        i += 1
      end
    end

    w = h[i].w
    v = h[i].v

    h[i].w = 0
    while i.positive?
      h[i].tw -= w
      i >>= 1
    end

    v
  end
end
