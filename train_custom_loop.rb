require 'numo/narray'
require_relative 'sgd'
require_relative 'spiral'
require_relative 'two_layer_net'

max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 0.01

x, t = load_data
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


