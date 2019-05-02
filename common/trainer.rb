require 'numo/narray'
require 'numo/gnuplot'

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

def remove_duplicates(params, grads)
  loop do
    found = false
    param_count = params.length

    param_count.times do |i|
      (i+1...param_count).each do |j|
        if params[i] == params[j]
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

  def fit(x, t, max_epoch: 0, batch_size: 32, max_grad: nil, eval_interval: 20)
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
      x = x[idx, true]
      t = t[idx, true]

      max_iters.times do |iters|
        batch_x = x[iters * batch_size...(iters+1) * batch_size, true]
        batch_t = t[iters * batch_size...(iters+1) * batch_size, true]

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

    Numo::gnuplot do
      set(yrange: ylim) if ylim
      set(terminal: 'png')
      set(output: 'loss_graph.png')
      set(key: 'box right top')
      set(xlabel: "iterations (x#{eval_interval})")
      set(ylabel: 'loss')
      plot(x, loss_list, w: 'lines', lw: 2, title: 'train')
    end
  end
end