# NNets as Deep GP Means

This repository aims to reproduce some results of the [Deep Neural Networks as Point Estimates for Deep Gaussian Processes](https://openreview.net/forum?id=svlanLvYsTd) paper by Dutordoir et al. (NeurIPS 2021).

The results here use `pytorch` and `gpytorch`.

Implementations available:
  - [x] Initializing single layer nnets using a fit GP.
  - [x] Initializing (single layer) GP using a fit nnet.
  - [ ] Initializing deep GPs using a fit deep nnet.
