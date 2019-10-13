# frozen_string_literal: true

require 'numo/narray'

class BaseModel
  attr_accessor :params, :grads

  def initialize
    @params = nil
    @grads = nil
  end

  def forward(*)
    raise NotImplementedError, "Please implemented 'forward'."
  end

  def backward(*)
    raise NotImplementedError, "Please implemented 'backward'."
  end

  def save_params(file_name = nil)
    fn = file_name || self.class.name + '.bin'

    File.open(fn, 'wb') do |f|
      Marshal.dump(params, f)
    end
  end

  def load_params(file_name = nil)
    fn = file_name || self.class.name + '.bin'
    fn = fn.gsub('/', File::SEPARATOR) if fn.include?('/')

    raise IOError, "No file: #{fn}" unless File.exist?(fn)

    params = nil
    File.open(fn, 'rb') do |f|
      params = Marshal.load(f)
    end

    @params.each_with_index do |param, i|
      param[] = params[i]
    end
  end
end
