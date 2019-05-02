require_relative '../common/optimizer'
require_relative '../common/trainer'
require_relative '../dataset/spiral'
require_relative 'two_layer_net'

max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = load_data
model = TwoLayerNet.new(2, hidden_size, 3)
optimizer = SGD.new(learning_rate)

trainer = Trainer.new(model, optimizer)
trainer.fit(x, t, max_epoch: max_epoch, batch_size: batch_size, eval_interval: 10)
trainer.plot
