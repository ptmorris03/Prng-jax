# Prng-jax
Pseudo-RNG convenience wrapper for quick experimentation in Jax. Initialize random tensors from a seed without thinking about psuedorandom state.

# Usage
```python3
from prng import Prng

seed = 0
prng = Prng(seed)

weights = prng(256, 512)
weights2 = prng(jax.random.normal, 3, 8, 4, 8)
weights3 = prng(jax.random.normal, (3, 2, 4, 8))
inputs = prng(jax.random.randint, (10, 50), 0, 1024)
scales = prng(jax.random.poisson, 3.0, (10, 50, 256))

print(weights.shape, weights2.shape, weights3.shape, inputs.shape, scales.shape)
```
