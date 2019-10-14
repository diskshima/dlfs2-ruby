# frozen_string_literal: true

require 'numo/narray'
require_relative '../ch07/seq2seq'
require_relative './attention_layer'

# Attention encoder
class AttentionEncoder < Encoder
  def forward(xs)
    xs = @embed.forward(xs)
    hs = @lstm.forward(xs)
    hs
  end

  def backward(dhs)
    dout = @lstm.backward(dhs)
    dout = @embed.backward(dout)
    dout
  end
end

# Attention decoder
class AttentionDecoder
  attr_accessor :params, :grads

  def initialize(vocab_size, wordvec_size, hidden_size)
    v = vocab_size
    d = wordvec_size
    h = hidden_size
    rn = ->(r, c) { Numo::SFloat.new(r, c).rand_norm }

    embed_w = rn.call(v, d) / 100
    lstm_wx = rn.call(d, 4 * h) / Numo::SFloat::Math.sqrt(d)
    lstm_wh = rn.call(h, 4 * h) / Numo::SFloat::Math.sqrt(h)
    lstm_b = Numo::SFloat.zeros(4 * h)
    affine_w = rn.call(2 * h, v) / Numo::SFloat::Math.sqrt(2 * h)
    affine_b = Numo::SFloat.zeros(v)

    @embed = TimeEmbedding.new(embed_w)
    @lstm = TimeLSTM.new(lstm_wx, lstm_wh, lstm_b, stateful: true)
    @attention = TimeAttention.new
    @affine = TimeAffine.new(affine_w, affine_b)
    layers = [@embed, @lstm, @attention, @affine]

    @params = []
    @grads = []
    layers.each do |layer|
      @params += layer.params
      @grads += layer.grads
    end
  end

  def forward(xs, enc_hs)
    h = enc_hs[true, -1]
    @lstm.set_state(h)

    out = @embed.forward(xs)
    dec_hs = @lstm.forward(out)
    c = @attention.forward(enc_hs, dec_hs)
    out = c.concatenate(dec_hs, axis: 2)
    score = @affine.forward(out)

    score
  end

  def backward(dscore)
    dout = @affine.backward(dscore)
    _n, _t, h2 = dout.shape
    h = h2 / 2

    dc = dout[true, true, 0...h]
    ddec_hs0 = dout[true, true, h..-1]
    denc_hs, ddec_hs1 = @attention.backward(dc)
    ddec_hs = ddec_hs0 + ddec_hs1
    dout = @lstm.backward(ddec_hs)
    dh = @lstm.dh
    denc_hs[true, -1] += dh
    @embed.backward(dout)

    denc_hs
  end

  def generate(enc_hs, start_id, sample_size)
    sampled = []
    sample_id = start_id
    h = enc_hs[true, -1]
    @lstm.set_state(h)

    sample_size.times do
      x = Numo::SFloat[[sample_id]]

      out = @embed.forward(x)
      dec_hs = @lstm.forward(out)
      c = @attention.forward(enc_hs, dec_hs)
      out = c.concatenate(dec_hs, axis: 2)
      score = @affine.forward(out)

      sample_id = score.max_index
      sampled.append(sample_id)
    end

    sampled
  end
end

# Sequence to sequence model using Attention
class AttentionSeq2seq < Seq2seq
  def initialize(vocab_size, wordvec_size, hidden_size)
    args = [vocab_size, wordvec_size, hidden_size]
    @encoder = AttentionEncoder.new(*args)
    @decoder = AttentionDecoder.new(*args)
    @softmax = TimeSoftmaxWithLoss.new

    @params = @encoder.params + @decoder.params
    @grads = @encoder.grads + @decoder.grads
  end
end
