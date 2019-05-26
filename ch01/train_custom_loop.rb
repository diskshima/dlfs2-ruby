require 'gnuplot'
require 'numo/narray'
require_relative '../common/layers'
require_relative '../common/optimizer'
require_relative '../dataset/spiral'
require_relative 'two_layer_net'

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
Gnuplot.open do |gp|
  Gnuplot::Plot.new(gp) do |plot|
    plot.data << Gnuplot::DataSet.new(loss_list) do |ds|
      ds.with = 'lines'
    end
  end
end

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

x_in = Numo::DFloat[*grid]
x_in *= h
score = model.predict(x_in)
predict_cls = argmax(score, axis: 1)


Gnuplot.open do |gp|
  Gnuplot::Plot.new(gp) do |plot|
    plots = []

    cls_num.times do |cls|
      idx = []
      predict_cls.eq(cls).each_with_index { |v, i| idx.push(v == 1 ? i : nil) }
      idx = idx.select { |v| !v.nil? }

      ds = Gnuplot::DataSet.new([x_in[idx, 0].to_a, x_in[idx, 1].to_a]) do |ds|
        ds.with = "points pt #{cls + 3}"
      end

      plots.append(ds)

      # Plot data points
      n = 100
      cls_num.times do |i|
        start_idx = i * n
        end_idx = (i + 1) * n

        ds = Gnuplot::DataSet.new(
          [x_orig[start_idx...end_idx, 0].to_a, x_orig[start_idx...end_idx, 1].to_a]
        ) do |ds|
          ds.with = "points pt #{i + 1}"
          ds.linewidth = 3
        end

        plots.append(ds)
      end
    end

    plot.data = plots
  end
end
