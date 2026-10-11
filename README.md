# Decorating

## Deep Conceptors for Temporal Data Mining

With this project, we provide the software for LRNNs (linear recurrent 
neural networks). The software has been implemented by the authors in 
Python and the scientific programming language Octave (see [octave.org](http://www.octave.org), 
version 5.2.0). It has been tested under Linux. The main routines can 
be found in `lrnn` implemented in Octave. For further details of LRNNs 
and their properties, the interested reader is referred to the 
following paper.

The self-contained [Python implementation](python/README.md) includes the
autonomous LRNN library, spectral dimension reduction, tests, a runnable
example and reproducible MSO/Python–Octave comparisons. Its
[accuracy guide](python/docs/ACCURACY.md) explains the fitting settings and
measured forecast accuracy. Python sources retain their MIT licence; the
original Octave sources retain their BSD 3-Clause licence.

## Attributions

Frieder Stolzenburg, Sandra Litz, Olivia Michael and Oliver Obst.
[Efficient time-series approximation with linear recurrent neural networks:
architecture learning and predictive power](https://link.springer.com/article/10.1007/s00521-025-11655-y).
*Neural Computing and Applications* **37**, 27027–27055 (2025).
DOI: `10.1007/s00521-025-11655-y`.


## Acknowledgements

The research reported here has been supported by the German Academic Exchange
Service (DAAD) by funds of the German Federal Ministry of Education and Research
(BMBF) in the Programmes for Project-Related Personal Exchange (PPP) under grant
no. 57319564 and Universities Australia (UA) in the Australia-Germany Joint
Research Cooperation Scheme within the project Deep Conceptors for Temporal Data
Mining (Decorating).


## Licenses

The original Octave code in the Decorating project is released under
[BSD-3-Clause License](https://opensource.org/license/bsd-3-clause/): https://opensource.org/license/bsd-3-clause/
The self-contained Python implementation retains its [MIT licence](python/LICENSE).

