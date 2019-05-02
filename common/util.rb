require 'numo/narray'

def clip_grads(grads, max_norm)
  total_norm = grads.reduce(0) do |total, grad|
    total + (grad ** 2).sum
  end
  total_norm = Numo::DFloat::Math.sqrt(total_norm)

  rate = max_norm / (total_norml + 1e-6)

  if rate < 1
    grads.each { |grad| grads *= rate }
  end
end
