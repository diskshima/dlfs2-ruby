# frozen_string_literal: true

require 'numo/narray'
require_relative '../common/base_model'
require_relative '../common/time_layers'
require_relative '../common/functions'

class Encoder
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

    @embed = TimeEmbedding.new(embed_w)
    @lstm = TimeLSTM.new(lstm_wx, lstm_wh, lstm_b, stateful: false)

    @params = @embed.params + @lstm.params
    @grads = @embed.grads + @lstm.grads
    @hs = nil
  end

  def forward(xs)
    xs = @embed.forward(xs)
    hs = @lstm.forward(xs)
    @hs = hs
    hs[true, -1, true] # All batches (true), last hidden layer(-1, true)
  end

  def backward(dh)
    dhs = Numo::SFloat.zeros(@hs)
    dhs[true, -1, true] = dh

    dout = @lstm.backward(dhs)
    dout = @embed.backward(dout)
    dout
  end
end

class Decoder
  def initialize(vocab_size, wordvec_size, hidden_size)
    v = vocab_size
    d = wordvec_size
    h = hidden_size
    rn = ->(r, c) { Numo::SFloat.new(r, c).rand_norm }

    embed_w = rn.call(v, d)
    lstm_wx = rn.call(d, 4 * h) / Numo::SFloat::Math.sqrt(d)
    lstm_wh = rn.call(h, 4 * h) / Numo::SFloat::Math.sqrt(h)
    lstm_b = Numo::SFloat.zeros(4 * h)
    affine_w = rn(h, v) / Numo::SFloat::Math.sqrt(h)
    affine_b = Numo::SFloat.zeros(v)

    @embed = TimeEmbedding.new(embed_w)
    @lstm = TimeLSTM.new(lstm_wx, lstm_wh, lstm_b, stateful: true)
    @affine = TimeAffine.new(affine_w, affine_b)

    @params = []
    @grads = []
    (@embed + @lstm + @affine).each do |layer|
      @params += layer.params
      @grads += layer.grads
    end
  end

  def forward(xs, h)
    @lstm.set_state(h)

    out = @embed.forward(xs)
    out = @lstm.forward(out)
    score = @affine.forward(out)
    score
  end

  def backward(dscore)
    dout = @affine.backward(dscore)
    dout = @lstm.backward(dout)
    dout = @embed.backward(dout)
    dh = @lstm.dh
    dh
  end

  def generate(h, start_id, sample_size)
    sampled = []
    sample_id = start_id
    @lstm.set_state(h)

    sample_size.times do
      x = Numo::SFloat[[sample_id]]
      out = @embed.forward(x)
      out = @lstm.forward(out)
      score = @affine.forward(out)

      sample_id = score.max_index
      sampled.append(sample_id)
    end

    sampled
  end
end

class Seq2seq < BaseModel
  def initialize(vocab_size, wordvec_size, hidden_size)
    v = vocab_size
    d = wordvec_size
    h = hidden_size

    @encoder = Encoder.new(v, d, h)
    @decoder = Decoder.new(v, d, h)
    @softmax = TimeSoftmaxWithLoss.new

    @params = @encoder.params + @decoder.params
    @grads = @encoder.grads + @decoder.grads
  end

  def forward(xs, ts)
    decoder_xs = ts[true, 0...-1]
    decoder_ts = ts[true, 1..-1]

    h = @encoder.forward(xs)
    score = @decoder.forward(decoder_xs, h)
    loss = @softmax.forward(score, decoder_ts)
    loss
  end

  def backward(dout = 1)
    dout = @softmax.backward(dout)
    dh = @decoder.backward(dout)
    dout = @encoder.backward(dh)
    dout
  end

  def generate(xs, start_id, sample_size)
    h = @encoder.forward(xs)
    sampled = @decoder.generate(h, start_id, sample_size)
    sampled
  end
end
