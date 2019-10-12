# frozen_string_literal: true

require 'numo/narray'
require_relative '../common/functions'
require_relative '../common/time_layers'
require_relative './seq2seq'

class PeekyDecoder
  attr_accessor :params, :grads

  def initialize(vocab_size, wordvec_size, hidden_size)
    v = vocab_size
    d = wordvec_size
    h = hidden_size
    rn = ->(r, c) { Numo::SFloat.new(r, c).rand_norm }

    embed_w = rn.call(v, d)
    lstm_wx = rn.call(h + d, 4 * h) / Numo::SFloat::Math.sqrt(h + d)
    lstm_wh = rn.call(h, 4 * h) / Numo::SFloat::Math.sqrt(h)
    lstm_b = Numo::SFloat.zeros(4 * h)
    affine_w = rn.call(h + h, v) / Numo::SFloat::Math.sqrt(h)
    affine_b = Numo::SFloat.zeros(v)

    @embed = TimeEmbedding.new(embed_w)
    @lstm = TimeLSTM.new(lstm_wx, lstm_wh, lstm_b, stateful: true)
    @affine = TimeAffine.new(affine_w, affine_b)

    @params = []
    @grads = []
    [@embed, @lstm, @affine].each do |layer|
      @params += layer.params
      @grads += layer.grads
    end
  end

  def forward(xs, h_in)
    _n, t = xs.shape
    n, h = h_in.shape

    @lstm.set_state(h_in)

    out = @embed.forward(xs)
    # TODO Assumes h's shape is (1, n) is `[[1,2,3,4]]`
    hs = h_in.repeat(t, axis: 0).reshape(n, t, h)
    out = hs.concatenate(out, axis: 2)

    out = @lstm.forward(out)
    out = hs.concatenate(out, axis: 2)

    score = @affine.forward(out)
    @cache = h
    score
  end

  def backward(dscore)
    h = @cache

    dout = @affine.backward(dscore)
    dout = dout[true, true, h..-1]
    dhs0 = dout[true, true, 0...h]
    dout = @lstm.backward(dout)
    dembed = dout[true, true, h..-1]
    dhs1 = dout[true, true, 0...h]
    @embed.backward(dembed)

    dhs = dhs0 + dhs1
    dh = @lstm.dh + dhs.sum(axis: 1)
    dh
  end

  def generate(h, start_id, sample_size)
    sampled = []
    char_id = start_id
    @lstm.set_state(h)

    h_count = h.shape[1]
    peeky_h = h.reshape(1, 1, h_count)

    sample_size.times do
      x = Numo::NArray[char_id].reshape(1, 1)
      out = @embed.forward(x)

      out = peeky_h.concatenate(out, axis: 2)
      out = @lstm.forward(out)
      out = peeky_h.concatenate(out, axis: 2)
      score = @affine.forward(out)

      char_id = argmax(score.flatten)
      sampled.append(char_id)
    end

    sampled
  end
end

class PeekySeq2seq < Seq2seq
  attr_accessor :params, :grads

  def initialize(vocab_size, wordvec_size, hidden_size)
    v = vocab_size
    d = wordvec_size
    h = hidden_size
    @encoder = Encoder.new(v, d, h)
    @decoder = PeekyDecoder.new(v, d, h)
    @softmax = TimeSoftmaxWithLoss.new

    @params = @encoder.params + @decoder.params
    @grads = @encoder.grads + @decoder.grads
  end
end
