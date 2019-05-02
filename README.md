# "Deep Learning from Scratch 2" with Ruby

A Ruby implementation based on the book [Deep Learning from Scratch 2](https://www.oreilly.co.jp/books/9784873118369/).

## Setup

```bash
bundle
```

## Running the Code

### Running the Custom Loop

```bash
bundle exec ruby train_custom_loop.rb
```

Gnuplot emits font errors when running this code and the library used ([ruby-numo/numo-gnuplot](https://github.com/ruby-numo/numo-gnuplot)) seems to error out so the code has been configured to output to a PNG file.
Please open the generated `loss_graph.png` and `classification.png` to see the outputs.

### Running the Trainer Class

```bash
bundle exec ruby train.rb
```
Please open the `loss_graph.png` file to see the output.
