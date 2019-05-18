require 'gnuplot'
require 'numo/narray'
require_relative 'functions'
require_relative 'layers'
require_relative 'util'

def remove_duplicates(params, grads)
  params = params.clone
  grads = grads.clone

  loop do
    found = false
    param_count = params.length

    param_count.times do |i|
      (i+1...param_count).each do |j|
        if params[i].object_id == params[j].object_id
          grads[i] += grads[j]
          found = true
          params.delete_at(j)
          grads.delete_at(j)
        elsif params[i].ndim == 2 && params[j].ndim == 2 &&
              params[i].transpose.shape == params[j].shape &&
              params[i].transpose == params[j]
          grads[i] += grads[j].transpose
          found = true
          params.delete_at(j)
          grads.delete_at(j)
        end

        break if found
      end

      break if found
    end

    break unless found
  end

  [params, grads]
end

class Trainer
  def initialize(model, optimizer)
    @model = model
    @optimizer = optimizer
    @loss_list = []
    @eval_interval = nil
    @current_epoch = 0
  end

  def fit(x, t, max_epoch = 0, batch_size = 32, max_grad: nil, eval_interval: 20)
    data_size = x.shape[0]
    max_iters = data_size / batch_size
    @eval_interval = eval_interval
    model = @model
    optimizer = @optimizer
    total_loss = 0
    loss_count = 0

    start_time = Time.now
    max_epoch.times do |epoch|
      idx = (0...data_size).to_a.shuffle
      x = get_at_dim_index(x, 0, idx)
      t = get_at_dim_index(t, 0, idx)

      max_iters.times do |iters|
        batches = iters * batch_size...(iters+1) * batch_size
        batch_x = get_at_dim_index(x, 0, batches)
        batch_t = get_at_dim_index(t, 0, batches)

        loss = model.forward(batch_x, batch_t)
        model.backward()
        params, grads = remove_duplicates(model.params, model.grads)

        clip_grads(grads, max_grad) if max_grad

        optimizer.update(params, grads)
        total_loss += loss
        loss_count += 1

        if eval_interval && (iters % eval_interval) == 0
          avg_loss = total_loss / loss_count
          elapsed_time = Time.now - start_time
          printf("| epoch %d | iter %d / %d | time %d[ms] | loss %.8f\n",
                 epoch + 1, iters + 1, max_iters, elapsed_time * 1_000, avg_loss)
          @loss_list.append(avg_loss)
          total_loss = 0
          loss_count = 0
        end
      end

      @current_epoch += 1
    end
  end

  def plot(ylim: nil)
    x = (0...@loss_list.length).to_a
    eval_interval = @eval_interval
    loss_list = @loss_list

    Gnuplot.open do |gp|
      Gnuplot::Plot.new(gp) do |plot|

        plot.set(:yrange, ylim) if ylim
        plot.set(:key, 'box right top')
        plot.xlabel("iterations (x#{eval_interval})")
        plot.ylabel('loss')

        plot.data << Gnuplot::DataSet.new([x, loss_list]) do |ds|
          ds.title = 'train'
          ds.with = 'lines'
          ds.linewidth = 2
        end
      end
    end
  end
end
