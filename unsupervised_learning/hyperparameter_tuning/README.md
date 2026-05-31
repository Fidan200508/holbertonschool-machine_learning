# 0. Initialize Gaussian Process

## Description
This project implements a noiseless 1D Gaussian Process using a Radial Basis Function (RBF) kernel.

## Requirements
- Python 3.9
- NumPy

## Files

### 0-gp.py
Contains the `GaussianProcess` class.

#### Class: `GaussianProcess`

##### Constructor
```python
GaussianProcess(X_init, Y_init, l=1, sigma_f=1)
