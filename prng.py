import jax

from numbers import Number
from typing import Any


class Prng:
    """
    A pseudo-random number generator class.

    Args:
        seed (jax.typing.ArrayLike): The seed value for the random number generator.
        f (callable, optional): The default random number generation function. Defaults to jax.random.normal.

    Attributes:
        f (callable): The random number generation function.
        key (jax.Array): The PRNGKey used for generating random numbers.

    Methods:
        split(): Splits the PRNGKey and returns a new key.
        __call__(*args): Generates random numbers using the default function and the split key.

    """

    def __init__(
        self, seed: jax.typing.ArrayLike, f=jax.random.normal
    ) -> None:
        self.f = f

        # Initialize seed as a Jax PRNGKey Array
        if not isinstance(seed, jax.Array):
            seed = jax.random.PRNGKey(seed)
            
        self.key = seed

    def split(self) -> jax.Array:
        """
        Splits the PRNGKey and returns a new key.

        Returns:
            jax.Array: The new PRNGKey.

        """
        # update internal key and generate a new one
        self.key, new_key = jax.random.split(self.key)
        
        return new_key

    def _is_num(self, x: Any) -> bool:
        """
        Checks if the argument is a numeric type.
      
        This function uses the `numbers.Number` class to check for a wide range
        of numeric types, including integers, floats, and complex numbers.
      
        Args:
            self: Reference to the Prng class instance. (Usually omitted in docstrings)
            x: The argument to check for numeric type.
      
        Returns:
            bool: True if the argument is a numeric type, False otherwise.
        """
        return isinstance(x, Number)

    def __call__(self, *args) -> jax.Array:
        """
        Generates random numbers using the default function and the split key.
        If the first argument is callable, it it used as the function instead.

        Args:
            *args: Variable-length argument list.

        Returns:
            jax.Array: The generated random numbers.

        """
        f = self.f

        # no arguments given
        if len(args) == 0:
            args = (1,)

        # first argument is a function
        if callable(args[0]):
            f = args[0]
            args = args[1:] if len(args) > 1 else ()

        # pack numeric arguments as one shape
        if all(map(self._is_num, args)):
            return f(self.split(), args)

        # pass arguments directly
        return f(self.split(), *args)


if __name__ == "__main__":
    SEED = 0
    prng = Prng(SEED)
    weights = prng(256, 512)
    weights2 = prng(jax.random.normal, 3, 8, 4, 8)
    weights3 = prng(jax.random.normal, (3, 2, 4, 8))
    inputs = prng(jax.random.randint, (10, 50), 0, 1024)
    scales = prng(jax.random.poisson, 3.0, (10, 50, 256))
    print(weights.shape, weights2.shape, weights3.shape, inputs.shape, scales.shape)
