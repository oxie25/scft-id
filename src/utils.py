import math
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, TypeVar

import jax
import jax.numpy as jnp

# The number of sequence points needs to be defined globally to enable lax.scan of for loops
GLOBAL_NS = 125  # For checkpointing, this cannot be a prime number. For Simpson integration, this must be an odd number.

# Automatically resolve the project root relative to this file
PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)  # adjust depth based on file's location

# Define where the STRUCTURES_DIR is relative to the project root, or provide its absolute path
STRUCTURES_DIR = PROJECT_ROOT / "structures"  # Directory for structures
STRUCTURES_DIR = r'D:\Oliver\for_jax'

# Define where the saved structures are going
CONVERGED_STRUCTURES_DIR = (
    PROJECT_ROOT / "converged_structures"
)  # Directory for converged structures

# Define where the figures are going
FIGURES_DIR = PROJECT_ROOT / "figures"  # Directory for figures

# Define where the initial guess structures are stored
INITIAL_GUESS_DIR = PROJECT_ROOT / "initial_guess"  # For initial guesses

# Code slightly modified from: https://github.com/jax-ml/jax/issues/2139
# Checkpointing is a technique to reduce memory usage during backpropagation
CHECKPOINT_FLAG = False
NESTED_LENGTHS = (25, 5) # This must multiply to GLOBAL_NS

Carry = TypeVar("Carry")
Input = TypeVar("Input")
Output = TypeVar("Output")
Func = TypeVar("Func", bound=Callable)


def nested_checkpoint_scan(
    f: Callable[[Carry, Input], Tuple[Carry, Output]],
    init: Carry,
    xs: Input,
    length: Optional[int] = None,
    *,
    nested_lengths: Sequence[int],
    checkpoint_fn: Callable[[Func], Func] = jax.checkpoint,
) -> Tuple[Carry, Output]:
    """A version of lax.scan that supports recursive gradient checkpointing.

    The interface of `nested_checkpoint_scan` exactly matches lax.scan, except for
    the required `nested_lengths` argument.

    The key feature of `nested_checkpoint_scan` is that gradient calculations
    require O(max(nested_lengths)) memory, vs O(prod(nested_lengths)) for unnested
    scans, which it achieves by re-evaluating the forward pass
    `len(nested_lengths) - 1` times.

    `nested_checkpoint_scan` reduces to `lax.scan` when `nested_lengths` has a
    single element.

    Args:
      f: function to scan over.
      init: initial value.
      xs: scanned over values.
      length: leading length of all dimensions
      nested_lengths: required list of lengths to scan over for each level of
        checkpointing. The product of nested_lengths must match length (if
        provided) and the size of the leading axis for all arrays in ``xs``.
      checkpoint_fn: function matching the API of jax.checkpoint.

    Returns:
      Carry and output values.
    """

    if length is not None:
        if length != math.prod(nested_lengths):
            raise ValueError(f"inconsistent {length=} and {nested_lengths=}")
    else:
        if xs.shape[0] != math.prod(nested_lengths):
            raise ValueError(
                f"inconsistent xs.shape[0]={xs.shape[0]} and {nested_lengths=}"
            )

    def nested_reshape(x):
        x = jnp.asarray(x)
        new_shape = tuple(nested_lengths) + x.shape[1:]
        return x.reshape(new_shape)

    sub_xs = jax.tree_util.tree_map(nested_reshape, xs)
    return _inner_nested_scan(
        f, init, sub_xs, nested_lengths, jax.lax.scan, checkpoint_fn
    )


def _inner_nested_scan(f, init, xs, lengths, scan_fn, checkpoint_fn):
    """Recursively applied scan function."""
    if len(lengths) == 1:
        return scan_fn(f, init, xs, lengths[0])

    @checkpoint_fn
    def sub_scans(carry, xs):
        return _inner_nested_scan(f, carry, xs, lengths[1:], scan_fn, checkpoint_fn)

    carry, out = scan_fn(sub_scans, init, xs, lengths[0])
    stacked_out = jax.tree_util.tree_map(jnp.concatenate, out)
    return carry, stacked_out
