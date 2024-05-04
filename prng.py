import jax

from typing import Any


def is_num(arg: Any) -> bool:
    return isinstance(arg, (int, float))


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
        If the first argument is callable, it it used as the function instead.

        Args:
            *args: Variable-length argument list.

        Returns:
            jax.Array: The generated random numbers.

        """
        f = self.default_fn

        # no arguments given
        if len(args) == 0:
            args = (1,)

        # first argument is a function
        if callable(args[0]):
            f = args[0]
            args = args[1:]

        # pack numeric arguments as one shape
        if all(map(is_num, args)):
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
