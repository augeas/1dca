#1DCA
##Take Your Brain to the 1st Dimension

One-dimensional cellular automata in [Marimo](https://marimo.io/) and [Numpy](https://numpy.org/).
Rules are specified as (mostly) boolean expressions in terms of neighbourhoods of five cells, for
instance `(L^c)|(r^c)`.

Each token in a rule is a single character:

* variables: spatial, `L`, `l`, `c`, `r`, `R` and `p` (previous), 0 or 1
* constants: single digits 0-9
* integer operators: `+`, `-`, `*`, `%` (modular division)
* boolean operators: `&`, (and) `|`, (or) `^`, (xor) `<`, `=`, `>`
* parentheses: `()`
* predicate function: `?` (`?(predicate, expression-if-true, expression-if-false)`)

You can play with the notebook rendered in web assemby at [augeas.github.io/1dca/](https://augeas.github.io/1dca/).
Alternatively, create a virtualenv, clone the repo and install the dependencies:

```bash
mkdir ca_venv
python3 -m venv ca_venv
source ./ca_venv/bin/activate
git clone git@github.com:augeas/1dca.git
cd 1cda
pip install -r requirements.txt
marimo edit
```
