# Genetic Algorithm — Ackley Function Optimization (2D)

A Genetic Algorithm (GA) implementation in Python that finds the global
minimum of the 2D **Ackley function**, a classic multimodal benchmark
function used in evolutionary computation.

## Problem

Minimize:

```
f(x1, x2) = -20 * exp(-0.2 * sqrt(0.5 * (x1² + x2²)))
            - exp(0.5 * (cos(2πx1) + cos(2πx2)))
            + 20 + e
```

- **Search space:** -5 ≤ x1, x2 ≤ 5
- **Global minimum:** f(x\*) = 0 at x\* = (0, 0)

The Ackley function has many local minima, making it a good stress
test for optimization/search algorithms.

## GA Design

| Component        | Choice |
|-------------------|--------|
| Encoding          | Value (real-number) encoding — chromosome = `(x1, x2)` |
| Population size   | 50 |
| Generations       | 100 |
| Selection         | Roulette wheel selection |
| Crossover         | One-point crossover (single possible cut point for a 2-gene chromosome) |
| Crossover prob.   | 0.80 |
| Mutation          | Gaussian perturbation per gene, clipped to `[-5, 5]` |
| Mutation prob.    | 0.05 |
| Elitism           | 1 (best individual preserved every generation) |

Since the GA maximizes fitness but the Ackley function is a
**minimization** target, fitness is computed as:

```
fitness = 1 / (1 + f(x))
```

so a smaller Ackley value yields a larger fitness score (and a bigger
slice of the roulette wheel).

## Files

- `ackley_ga.py` — full GA implementation, entry point
- `ga_ackley_convergence.png` — example convergence plot (best fitness per generation)

## Requirements

- Python 3.8+
- `numpy`
- `matplotlib`

Install with:

```bash
pip install numpy matplotlib
```

## Usage

```bash
python ackley_ga.py
```

This prints progress every 10 generations, reports the best solution
found, and saves a convergence plot to `ga_ackley_convergence.png`.

Example output:

```
Generation   1 | Best f(x) so far = 3.763375 | x = (0.2273, -0.7246)
Generation  50 | Best f(x) so far = 0.108816 | x = (0.0174, -0.0245)
Generation 100 | Best f(x) so far = 0.025708 | x = (-0.0034, 0.0077)

===== RESULT =====
Best solution : x1 = -0.003354, x2 = 0.007725
Ackley f(x)   : 0.025708
Known global minimum: f(0, 0) = 0
```

## Notes

- A fixed random seed (`np.random.seed(42)`) is set for reproducibility.
  Remove or change it in `ackley_ga.py` to get different runs.
- Tune `MUTATION_STRENGTH`, `PC`, `PM`, `POP_SIZE`, or `GENERATIONS`
  in `ackley_ga.py` to experiment with convergence speed/quality.

## Contributors

- Ishrat Binte Ahmed, Roll: 2107019
- Adiba Tahsin, Roll: 2107031
- Shahriar Aziz Khan, Roll: 2107034
- Ayesha Mehereen, Roll: 2107039
- Megha Tania, Roll: 2107057

## License

MIT.
