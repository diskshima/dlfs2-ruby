require 'numo/narray'
require 'numo/gnuplot'
require_relative 'sgd'
require_relative 'spiral'
require_relative 'two_layer_net'
require_relative 'softmax_with_loss'

max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = load_data
x_orig = x
t_orig = t
model = TwoLayerNet.new(2, hidden_size, 3)
optimizer = SGD.new(learning_rate)

data_size = x.shape[0]
max_iters = data_size / batch_size
total_loss = 0
loss_count = 0
loss_list = []

max_epoch.times do |epoch|
  idx = (0...data_size).to_a.shuffle
  x = x[idx, true]
  t = t[idx, true]

  max_iters.times do |iters|
    batch_x = x[iters * batch_size...(iters+1) * batch_size, true]
    batch_t = t[iters * batch_size...(iters+1) * batch_size, true]

    loss = model.forward(batch_x, batch_t)
    model.backward()
    optimizer.update(model.params, model.grads)

    total_loss += loss
    loss_count += 1

    if (iters + 1) % 10 == 0
      avg_loss = total_loss / loss_count
      printf("| epoch %d | iter %d / %d | loss %.8f\n",
             epoch + 1, iters + 1, max_iters, avg_loss)
      loss_list.append(avg_loss)
      total_loss = 0
      loss_count = 0
    end
  end
end

# Plot loss history
gp_loss = Numo::Gnuplot.new
gp_loss.set(terminal: 'png')
gp_loss.set(output: 'loss_graph.png')
gp_loss.plot(loss_list)

gp_cls = Numo::Gnuplot.new
gp_cls.set(terminal: 'png')
gp_cls.set(output: 'classification.png')

cls_num = 3
h = 0.05
r = (1 / h).to_i
side_points = (-r...r).to_a
grid = []
side_points.each do |i|
  side_points.each do |j|
    grid.append([i, j])
  end
end

x_in = Numo::DFloat.new(side_points.length**2, 2).seq
x_in[] = grid
x_in.inplace * h
score = model.predict(x_in)
predict_cls = argmax(score)

plots = []

cls_num.times do |cls|
  idx = []
  predict_cls.eq(cls).each_with_index { |v, i| idx.push(v == 1 ? i : nil) }
  idx = idx.select { |v| !v.nil? }
  plots.append([x_in[idx, 0], x_in[idx, 1], w: 'points', pt: cls + 4])
end

# Plot data points
n = 100
cls_num.times do |i|
  start_idx = i * n
  end_idx = (i + 1) * n

  plots.append([x_orig[start_idx...end_idx, 0], x_orig[start_idx...end_idx, 1],
                w: 'points', pt: i + 1])
end

gp_cls.plot(*plots)
