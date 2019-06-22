require 'numo/narray'

dw1 = Numo::DFloat.new(3, 3).rand * 10
dw2 = Numo::DFloat.new(3, 3).rand * 10
grads = [dw1, dw2]
max_norm = 5.0

def clip_grads(grads, max_norm)
  total_norm = 0
  grads.each do |grad|
    total_norm += (grad ** 2).sum
  end
  total_norm = Math.sqrt(total_norm)

  rate = max_norm / (total_norm + 1e-6)

  if rate < 1
    grads.each do |grad|
      grad.inplace * rate
    end
  end
end

puts "Before: #{dw1.flatten.inspect}"
clip_grads(grads, max_norm)
puts "After:  #{dw1.flatten.inspect}"
