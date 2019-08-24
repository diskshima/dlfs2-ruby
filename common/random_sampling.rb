# frozen_string_literal: true

# Implementation of Weighted Random Sampling.
# Proudly stolen from:
#   https://stackoverflow.com/a/2149533/4037

Node = Struct.new(:w, :v, :tw)
Rand = Random.new

def rws_heap(items)
  h = [nil]
  items.each do |w, v|
    h.append(Node.new(w, v, w))
  end

  (h.length - 1).downto(2).each do |i|
    h[i >> 1].tw += h[i].tw
  end

  h
end

def rws_heap_pop(h)
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

def random_weighted_sample_no_replacement(items, n)
  heap = rws_heap(items)
  n.times do
    yield rws_heap_pop(heap)
  end
end
