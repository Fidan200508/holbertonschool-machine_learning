# Advanced Linear Algebra

## Project Description
This project implements advanced linear algebra algorithms in Python without using external libraries. These implementations are foundational for understanding the mechanics behind AI architectures and industrial robotics control systems.

## Tasks

### 0. Determinant
A function `def determinant(matrix):` that calculates the determinant of a square matrix.

- **Input**: A list of lists representing a square matrix.
- **Output**: The determinant of the matrix.
- **Edge Cases**: Handles `0x0` as `1`, and raises appropriate `TypeError` or `ValueError` for invalid inputs.

## Usage
```bash
determinant = __import__("0-determinant").determinant
mat = [[5, 7], [3, 1]]
print(determinant(mat))  # Output: -16
```
