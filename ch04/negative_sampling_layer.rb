require 'numo/narray'
require_relative '../common/layers'

class EmbeddingDot
  attr_accessor :params, :grads

  def initialize(w)
    @embed = Embedding.new(w)
    @params = @embed.params
    @grads = @embed.grads
    @cache = nil
  end

  def forward(h, idx)
    target_w = @embed.forward(idx)
    out = (target_w * h).sum(axis: 1)
    @cache = [h, target_w]
    out
  end

  def backward(dout)
    h, target_w = @cache
    dout = dout.reshape(dout.shape[0], 1)

    dtarget_w = dout * h
    @embed.backward(dtarget_w)
    dh = dout * target_w
    dh
  end
end

class UnigramSampler
  def initialize(corpus, power, sample_size)
    @sample_size = sample_size
    @vocab_size = nil
    @word_p = nil

    counts = {}

    corpus.each do |word_id|
      counts[word_id] = counts[word_id] ? counts[word_id] + 1 : 1
    end

    vocab_size = counts.length
    @vocab_size = vocab_size

    @word_p = Numo::SFloat.zeros(vocab_size)

    vocab_size.times do |i|
      @word_p[i] = counts[i]
    end

    @word_p = @word_p ** power
    @word_p /= @word_p.sum
  end

  def get_negative_sample(target)
    batch_size = target.shape[0]

    negative_sample = Numo::Int32.zeros(batch_size, @sample_size)

    batch_size.times do |i|
      p = @word_p.copy
      target_idx = target[i]
      p[target_idx] = 0
      p /= p.sum
      negative_sample[i, true] =
        random_choice(@vocab_size, size: @sample_size, p: p)
    end

    negative_sample
  end
end

class NegativeSamplingLoss
  attr_accessor :params, :grads

  def initialize(w, corpus, power: 0.75, sample_size: 5)
    @sample_size = sample_size
    @sampler = UnigramSampler.new(corpus, power, sample_size)
    @loss_layers = (sample_size + 1).times.map { SigmoidWithLoss.new }
    @embed_dot_layers = (sample_size + 1).times.map { EmbeddingDot.new(w) }

    @params = []
    @grads = []

    @embed_dot_layers.each do |layer|
      @params += layer.params
      @grads += layer.grads
    end
  end

  def forward(h, target)
    batch_size = target.shape[0]
    negative_sample = @sampler.get_negative_sample(target)

    score = @embed_dot_layers[0].forward(h, target)
    correct_label = Numo::Int32.ones(batch_size)
    loss = @loss_layers[0].forward(score, correct_label)

    negative_label = Numo::Int32.zeros(batch_size)
    @sample_size.times do |i|
      negative_target = negative_sample[true, i]
      score = @embed_dot_layers[1 + i].forward(h, negative_target)
      loss += @loss_layers[1 + i].forward(score, negative_label)
    end

    loss
  end

  def backward(dout = 1.0)
    dh = 0

    @loss_layers.zip(@embed_dot_layers).each do |l0, l1|
      dscore = l0.backward(dout)
      dh += l1.backward(dscore)
    end

    dh
  end
end
