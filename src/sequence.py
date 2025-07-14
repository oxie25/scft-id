import jax.numpy as jnp
import equinox as eqx
from typing import Callable

from . import sequence_utils
from .scft_utils import simpsons_weights


class Sequence(eqx.Module):
    name: str
    basis: str
    transform: str
    basis_fun: Callable
    transform_fun: Callable
    f_params: jnp.ndarray
    Nbasis: int
    Ns: int
    Np: int
    Ndim: int
    dt: float
    stencil: jnp.ndarray

    @classmethod
    def generate(
        cls,
        name: str,
        basis: str,
        transform: str,
        Nbasis: int,
        Np: int,
        Ndim: int,
        f_params: jnp.ndarray = jnp.array([]),
        Ns: int = 125,
    ):
        # The basis function. For multiblocks and supplied sequences, it is linear
        if not hasattr(sequence_utils, basis):
            raise ValueError(f"Basis function '{basis}' not found in sequence_utils.")
        basis_fun = getattr(sequence_utils, basis)
        # The transform function. For multiblocks, it is multiblock, for supplied sequences, it is no_transform
        if not hasattr(sequence_utils, transform):
            raise ValueError(
                f"Transform function '{transform}' not found in sequence_utils."
            )
        transform_fun = getattr(sequence_utils, transform)

        # Discretization
        dt = 1.0 / (Ns - 1)

        stencil = simpsons_weights(Ns).reshape(Ns, 1) * dt  # Shape (Ns, 1)

        return cls(
            name=name,
            basis=basis,
            transform=transform,
            basis_fun=basis_fun,
            transform_fun=transform_fun,
            f_params=f_params,
            Ns=Ns,
            Np=Np,
            Nbasis=Nbasis,
            Ndim=Ndim,
            dt=dt,
            stencil=stencil,
        )
