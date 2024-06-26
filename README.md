# jaxprng
Pseudo-RNG convenience wrapper for quick experimentation in Jax. Initialize random tensors from a seed without thinking about psuedorandom state.

# Usage
```python3
from jaxprng import Prng

seed = 0
prng = Prng(seed)

weights = prng(256, 512)                                 # jax.random.normal
weights2 = prng(jax.random.normal, 3, 8, 4, 8)           # (*shape,)
weights3 = prng(jax.random.normal, (3, 2, 4, 8))         # (shape,)
inputs = prng(jax.random.randint, (10, 50), 0, 1024)     # (shape, low, high)
scales = prng(jax.random.poisson, 3.0, (10, 50, 256))    # (lambda, shape)

print(weights.shape, weights2.shape, weights3.shape, inputs.shape, scales.shape)
```
