# Helper functions for constructing the sequence function representation of polymers
import jax
import jax.numpy as jnp

from .utils import GLOBAL_NS


# Different supported basis functions for sequence representation
# Linear basis function
def linear_basis(x, N):
    # This function returns the identity matrix
    return jnp.eye(N)


# Chebyshev basis function
def chebyshev_basis(x, N):
    # Transform x from [0, 1] to [-1, 1]
    x_transformed = 2 * x - 1

    # Create a list to hold the polynomials
    chebyshev_list = []

    # Initialize the first two Chebyshev polynomials
    T0 = jnp.ones_like(x_transformed)  # T0(x) = 1
    T1 = x_transformed  # T1(x) = x_transformed

    chebyshev_list.append(T0)
    chebyshev_list.append(T1)

    # Compute the remaining polynomials up to degree N
    for n in range(2, N):
        Tn = 2 * x_transformed * chebyshev_list[-1] - chebyshev_list[-2]
        chebyshev_list.append(Tn)

    return jnp.stack(chebyshev_list)


# Polynomial basis function
def polynomial_basis(x, N):
    # Transform x from [0, 1] to [0, 1]
    x_transformed = 2 * x - 1

    # Create a list to hold the polynomials
    polynomial_list = []

    # Calculate the first polynomial
    T0 = jnp.ones_like(x_transformed)  # T0(x) = 1
    polynomial_list.append(T0)

    # Calculate the remaining polynomials up to degree N
    for n in range(1, N):
        Tn = x_transformed**n
        polynomial_list.append(Tn)

    return jnp.stack(polynomial_list)


# Calculation of the fractions
def calculate_fractions(logits):
    # 1) partition fractions
    f = jax.nn.softmax(logits)  # (D,)

    return f


# Different supported transformations for sequence representation
def no_transform(p):
    # No transformation, just return the input
    return p


def block_copolymer(p):
    # This is a base representation for a block copolymer where we specify exactly the fractions of each block and the interblock chi
    N = GLOBAL_NS
    f = p[0, :]  # First row is now f
    mag = p[1:, :]  # All rows except the first goes into the chemical identity
    # f: (D,), mag: (R, D)
    arr = jnp.zeros((N, mag.shape[0]))
    # Compute block boundaries
    b = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(f)])  # (D+1,)
    spts = jnp.linspace(0, 1, N)  # (N,)

    # For each block d, fill in the corresponding region for each row r
    for d in range(f.shape[0]):
        mask = (spts >= b[d]) & (spts < b[d + 1])
        for r in range(mag.shape[0]):
            arr = arr.at[mask, r].set(mag[r, d])
    # Ensure last point is included in the last block
    arr = arr.at[-1, :].set(mag[:, -1])
    return arr


def multiblock(p):
    # Limitation of 'block' was that it could not reproduce if two blocks were the same. This can, but at the expense of the identiy maybe going wonky.
    N = GLOBAL_NS
    f = p[0, :]  # First row is now f
    mag = p[1:, :]  # All rows except the first goes into the chemical identity

    spts = jnp.linspace(0, 1, N)[:, None]  # (N,1)

    # We need to use the sigmoid approximation, then sample at the discretized points
    block = multiblock_tophat(f, mag, spts)

    return block


def multiblock_taper(p):
    # Limitation of 'block' was that it could not reproduce if two blocks were the same. This can, but at the expense of the identiy maybe going wonky.
    N = GLOBAL_NS
    f = p[0, :]  # First row is now f
    slope = p[1, :]  # First element of second row is slope of the taper
    mag = p[2:, :]  # All rows except the first goes into the chemical identity

    spts = jnp.linspace(0, 1, N)[:, None]  # (N,1)

    # We need to use the sigmoid approximation, then sample at the discretized points
    block = multiblock_tophat_slope(f, mag, slope, spts)

    return block


def multiblock_tophat(logits, heights, x, sharpness=100.0):
    # 1) partition fractions
    f = jax.nn.softmax(logits)  # (D,)

    # 2) break-points in [0,1]
    b = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(f)])  # (D+1,)

    # 3) build smooth “rectangle” basis on each interval [b[d], b[d+1]]:
    #    left[d](x)  ≈ sigmoid( sharpness * (x - b[d]) )
    #    right[d](x) ≈ sigmoid( sharpness * (b[d+1] - x) )
    #    mask[:,d]   = left[d] * right[d]
    left = jax.nn.sigmoid(sharpness * (x - b[:-1][None, :]))  # (N,D)
    right = jax.nn.sigmoid(sharpness * (b[1:][None, :] - x))  # (N,D)

    # clamp the outermost edges to exactly 1
    left = left.at[:, 0].set(1.0)
    right = right.at[:, -1].set(1.0)

    mask = left * right  # (N,D)
    mask = mask / (mask.sum(axis=1, keepdims=True) + 1e-8)  # normalize to sum = 1

    # 4) weight-and-sum for each row of heights:
    #    profiles[r, n] = Σ_d  heights[r,d] * mask[n,d]
    #    => profiles = heights @ mask.T
    profiles = jnp.dot(heights, mask.T)  # (R, N)

    # Transpose profile to (N, R)
    profiles = jnp.transpose(profiles)  # (N, R)

    return profiles


def multiblock_tophat_slope(logits, heights, slopes, x, sharpness=100.0):
    f = jax.nn.softmax(logits)  # (D,)
    b = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(f)])  # (D+1,)

    left = jax.nn.sigmoid(sharpness * (x - b[:-1][None, :]))  # (N,D)
    right = jax.nn.sigmoid(sharpness * (b[1:][None, :] - x))  # (N,D)

    left = left.at[:, 0].set(1.0)
    right = right.at[:, -1].set(1.0)

    mask = left * right  # (N,D)
    mask = mask / (mask.sum(axis=1, keepdims=True) + 1e-8)

    # Compute linear offset for each block at each x
    x_minus_bd = x - b[:-1][None, :]  # (N,D)
    offset = slopes[None, :] * x_minus_bd  # (N,D) ← slopes broadcasted

    # Add offset to base heights
    # heights: (R, D), offset: (N, D)
    # Want final shape: (N, R, D)
    heights_exp = heights[None, :, :]  # (1, R, D)
    offset_exp = offset[:, None, :]  # (N, 1, D)
    values = heights_exp + offset_exp  # (N, R, D)

    # Apply mask
    mask_exp = mask[:, None, :]  # (N, 1, D)
    weighted = values * mask_exp  # (N, R, D)
    profiles = weighted.sum(axis=-1)  # (N, R)

    return profiles


# Calculate the basis coefficients for a desired sequence function f_desired in a given basis
def calculate_basis_coefficients(f_desired, basis):
    # Solve the least squares problem to find the Chebyshev coefficients
    f_basis, *_ = jnp.linalg.lstsq(basis.T, f_desired, rcond=None)
    return f_basis
