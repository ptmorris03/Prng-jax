import jax
import jax.numpy as jnp

from functools import partial


class Prng:
    """
    A pseudo-random number generator class.

    Args:
        seed (jax.typing.ArrayLike): The seed value for the random number generator.
        default_fn (callable, optional): The default random number generation function. Defaults to jax.random.normal.

    Attributes:
        default_fn (callable): The default random number generation function.
        key (jax.Array): The PRNGKey used for generating random numbers.

    Methods:
        split(): Splits the PRNGKey and returns a new key.
        __call__(*args): Generates random numbers using the default function and the split key.

    """

    def __init__(
        self, seed: jax.typing.ArrayLike, default_fn=jax.random.normal
    ) -> None:
        self.default_fn = default_fn
        if not isinstance(seed, jax.Array):
            seed = jax.random.PRNGKey(seed)
        self.key = seed

    def split(self) -> jax.Array:
        """
        Splits the PRNGKey and returns a new key.

        Returns:
            jax.Array: The new PRNGKey.

        """
        self.key, new_key = jax.random.split(self.key)
        return new_key

    def __call__(self, *args):
        """
        Generates random numbers using the default function and the split key.

        Args:
            *args: Variable-length argument list.

        Returns:
            jax.Array: The generated random numbers.

        """
        f = self.default_fn
        if len(args) == 0:
            shape = (1,)
        elif callable(args[0]):
            f = args[0]
            shape = args[1] if len(args) == 2 else args[1:] if len(args) > 1 else (1,)
        else:
            shape = args[0] if len(args) == 1 else args

        return f(self.split(), shape)


if __name__ == "__main__":
    seed = 0
    prng = Prng(seed)
    weights = prng(256, 512)
    weights2 = prng(jax.random.normal, 3, 8, 4, 8)
    weights3 = prng(jax.random.normal, (3, 2, 4, 8))
    inputs = prng(jax.random.randint, (10, 50), 0, 1024)
    scales = prng(jax.random.poisson, 3.0, (10, 50, 256))
    print(weights.shape, weights2.shape, weights3.shape, inputs.shape, scales.shape)
