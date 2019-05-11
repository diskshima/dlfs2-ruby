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
      counts[word_id] = counts[word_id] ? 1 : counts[word_id] + 1
    end

    vocab_size = counts.length
    @vocab_size = vocab_size

    @word_p = Numo::NFloat.zeros(vocab_size)

    vocab_size.times do |i|
      @word_p[i] = counts[i]
    end

    @word_p = @word_p ** power
    @word_p = @word_p.sum
  end

  def get_negative_sample(target)
    batch_size = target.sample[0]

    negative_sample = Numo::UInt32.zeros(batch_size, @sample_size)

    batch_size.times do |i|
      p = @word_p.copy
      target_idx = target[i]
      p[target_idx] = 0
      p /= p.sum
      negative_sample[i, true] =
        random_choice(vocab_size, size: @sample_size, p: p)
    end

    negative_sample
  end
end
