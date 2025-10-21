### Store the functions for SCFT here.

import jax
import jax.numpy as jnp
from jax.scipy import integrate

from .utils import nested_checkpoint_scan

jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision

frac_sigmoid_factor = 2  # Set globally to prevent forgetting it somewhere

Ns_static = 125


# Simpson's rule weights: for an odd number of intervals (even number of points)
def simpsons_weights(Ns):
    if Ns < 3 or Ns % 2 == 0:
        raise ValueError("Simpson's rule requires an odd number of samples.")

    weights = jnp.ones(Ns)
    weights = weights.at[1:-1:2].set(4)  # Assign 4 to odd indices
    weights = weights.at[2:-1:2].set(
        2
    )  # Assign 2 to even indices (between odd indices)
    weights = weights / 3  # Apply Simpson's rule factor (1/3)
    return weights


### The SCFT functions, with jax.jit to enable fast calculation and backwards propagation
@jax.jit
def integral(field, stencil):
    # We must enforce incompressibility
    # psi_solve is Ntau x Ns
    # stencil is Ns x 1
    result = jnp.matmul(field, stencil)
    return result


@jax.jit
def strang_splitting_scan(w, u_forward, u_backward, FT, FS, GsqStar, stencil):
    Ns = Ns_static
    dt = 1.0 / (Ns - 1)
    D = 1 / 6

    # Reshape w to have shape (time, space, 2) where second channel is reversed w
    w_forward = w.T  # Shape (time, space)
    w_backward = jnp.flip(w_forward, axis=0)  # Reversed in time dimension
    w_stack = jnp.stack([w_forward, w_backward], axis=-1)  # Shape (time, space, 2)

    diffusion_full = jnp.exp(-D * dt * GsqStar)

    potential_half = jnp.exp(-w_stack * dt / 2)

    def strang(state, potential_half_t):

        u_forward, u_backward = state

        # Forward in time integration
        u_step_tau_forward = potential_half_t[:, 0] * u_forward
        u_step_star_forward = jnp.matmul(FT, u_step_tau_forward)
        u_step_star_forward = diffusion_full * u_step_star_forward
        u_step_tau_forward = jnp.matmul(FS, u_step_star_forward)
        u_step_tau_forward = potential_half_t[:, 0] * u_step_tau_forward  # what we want

        u_step_tau_backward = potential_half_t[:, 1] * u_backward
        u_step_star_backward = jnp.matmul(FT, u_step_tau_backward)
        u_step_star_backward = diffusion_full * u_step_star_backward
        u_step_tau_backward = jnp.matmul(FS, u_step_star_backward)
        u_step_tau_backward = potential_half_t[:, 1] * u_step_tau_backward  # what we want

        return (u_step_tau_forward, u_step_tau_backward), (
            u_forward,
            u_backward,
        )

    initial_state = (u_forward, u_backward)

    #_, final_state = nested_checkpoint_scan(strang, initial_state, potential_half, nested_lengths=(25,5))
    _, final_state = jax.lax.scan(strang, initial_state, potential_half)

    # Unpack final forward and backward results
    u_final_forward, u_final_backward = final_state

    # Send back the integrated u_forward * u_backward in star space
    # Currently doesn't handle b(s)
    u_final_star_forward = jnp.matmul(FT, u_final_forward.T)
    u_final_star_backward = jnp.matmul(FT, u_final_backward.T)

    int_u_prod = integral(
        u_final_star_forward * jnp.flip(u_final_star_backward, 1), stencil
    )  # Flip the backward!

    return u_final_forward.T, u_final_backward.T, int_u_prod


@jax.jit
def strang_splitting(w, u_forward, u_backward, FT, FS, GsqStar, stencil):
    # Modified to allow block asymmetry - TODO
    Ns = Ns_static
    dt = 1.0 / (Ns - 1)
    D = 1 / 6
    Ntaus = w.shape[0]  # Number of spatial points
    u_forward = jnp.zeros((Ntaus, Ns)).at[:, 0].set(u_forward)
    u_backward = jnp.zeros((Ntaus, Ns)).at[:, 0].set(u_backward)

    diffusion_full = jnp.exp(-D * dt * GsqStar)

    potential_half = jnp.exp(-w * dt / 2)

    def strang(n, state):

        u_forward, u_backward = state

        # Forward in time integration
        u_step_tau_forward = potential_half[:, n] * u_forward[:, n]
        u_step_star_forward = jnp.matmul(FT, u_step_tau_forward)
        u_step_star_forward = diffusion_full * u_step_star_forward
        u_step_tau_forward = jnp.matmul(FS, u_step_star_forward)
        u_step_tau_forward = potential_half[:, n] * u_step_tau_forward

        # Backward in time integration
        # For backwards, flip `w` by indexing it as `w[:, num_steps - n - 1]`
        backward_idx = Ns - n - 1
        u_step_tau_backward = (
            potential_half[:, backward_idx.astype(int)] * u_backward[:, n]
        )
        u_step_star_backward = jnp.matmul(FT, u_step_tau_backward)
        u_step_star_backward = diffusion_full * u_step_star_backward
        u_step_tau_backward = jnp.matmul(FS, u_step_star_backward)
        u_step_tau_backward = (
            potential_half[:, backward_idx.astype(int)] * u_step_tau_backward
        )

        # Update forward and backward states
        u_forward = u_forward.at[:, n + 1].set(u_step_tau_forward)
        u_backward = u_backward.at[:, n + 1].set(u_step_tau_backward)

        return u_forward, u_backward

    initial_state = (u_forward, u_backward)

    final_state = jax.lax.fori_loop(int(0), Ns, strang, initial_state)
    # TODO: CHECKPOINT THIS FOR LOOP

    # Unpack final forward and backward results
    u_final_forward, u_final_backward = final_state

    # Send back the integrated u_forward * u_backward in star space
    # Currently doesn't handle b(s)
    u_final_star_forward = jnp.matmul(FT, u_final_forward)
    u_final_star_backward = jnp.matmul(FT, u_final_backward)

    int_u_prod = integral(
        u_final_star_forward * jnp.flip(u_final_star_backward, 1), stencil
    )  # Flip the backward!

    #print("int_u_prod shape:", int_u_prod.shape)
    #print("u_final_forward shape:", u_final_forward.shape)
    #print("u_final_backward shape:", u_final_backward.shape)
    #jax.debug.print("{u_final_forward}", u_final_forward=u_final_forward)

    return u_final_forward, u_final_backward, int_u_prod


@jax.jit
def pairwise(f):
    # Compute the pairwise 2-norm differences using broadcasting
    # f[:, None] expands f to shape (Ns, 1, Nx) and f[None, :] expands it to (1, Ns, Nx)
    # The result will be of shape (Ns, Ns, Nx)
    diff = f[:, None, :] - f[None, :, :]

    # Currently cannot implement Euclidean distance because sqrt operator not differentiable for diagonal (where all zero)
    interact_matrix = jnp.mean(
        diff**2, axis=-1
    )  # Take average to prevent explosion of size

    return interact_matrix


@jax.jit
def psi_solve(w, u_forward, u_backward, dt, Ns, FT, FS, GsqStar, stencil):
    # Find the density field psi
    u_final_forward, u_final_backward, int_u_prod = strang_splitting_scan(
        w, u_forward, u_backward, FT, FS, GsqStar, stencil
    )

    # Q is the integral over space of u_final_forward at last position
    # We need to average over the forward and reverse solves to built the autodiff graph appropriately
    u_final_forward_star = jnp.matmul(FT, u_final_forward[:, -1])
    Q_fwd = u_final_forward_star[
        0
    ]  # First element of star is a good approximate for integral

    u_final_backward_star = jnp.matmul(FT, u_final_backward[:, -1])
    Q_rev = u_final_backward_star[
        0
    ]  # First element of star is a good approximate for integral

    Q = (Q_fwd + Q_rev) / 2

    psi = (
        u_final_forward * jnp.flip(u_final_backward, 1) / Q
    )  # We need to flip u_final_backward since it is from 1 to 0!

    return psi, Q, int_u_prod


@jax.jit
def sequence_to_identity(f, dim=0, crossover=0, width=100):
    # Changes a sequence function (Ns x Np) to an identity function (Ns x Ni)
    # Large width makes curves steeper, small width makes curves wider
    g = jax.nn.sigmoid(width * (f[:, dim] - crossover))

    return g


#@jax.jit
def hamiltonian(c, Q, Q_solv, psi, psi_solv, w, w_solv, L_ss, chi_solv, dt, FT, FS):
    # Free energy calculation using trapezoid rule integration scheme
    # Trapezoid rule returns same results as MATLAB, not simpson
    term_A = -c * jnp.log(Q) - (1 - c) * jnp.log(Q_solv)
    #wpsi_tau = jnp.matmul(psi * w, stencil) # Integration over s space
    wpsi = psi * w
    wpsi_tau = integrate.trapezoid(wpsi, dx=dt, axis=1)
    wpsi_star = jnp.matmul(FT, wpsi_tau)  # pseudo-integration in tau space
    # Calculate for solvent
    wpsi_solv = psi_solv * w_solv
    wpsi_solv_star = jnp.matmul(FT, wpsi_solv)
    
    term_B = -wpsi_star[0] - wpsi_solv_star[0]

    L_expanded = L_ss[None, :, :]
    psi_s_expanded = psi[:, :, None]
    psi_sprime_expanded = psi[:, None, :]
    Lpp = L_expanded * psi_s_expanded * psi_sprime_expanded

    # Use trapezoidal rule to integrate over s and s' because of numerical errors w/ MATLAB code
    Lpp_ints = integrate.trapezoid(Lpp, dx=dt, axis=2)
    Lpp_ints2 = integrate.trapezoid(Lpp_ints, dx=dt, axis=1)

    # Transform to star space for pseudo-integration
    Lpp_star = jnp.matmul(FT, Lpp_ints2)

    # Calculate the FH interaction contribution of the solvent
    F_solv = psi_solv.flatten() * integrate.trapezoid(chi_solv * psi, dx=dt, axis=1)  # Integrate over s (psi is dim Ntau x Ns)
    F_solv_star = jnp.matmul(FT, F_solv)

    term_C = 1 / 2 * Lpp_star[0] + 1 / 2 * F_solv_star[0]

    # Sum up energy terms
    energy = term_A + term_B + term_C

    return energy


@jax.jit
def Gbasis_calc(Rbasis):
    # Access each ROW of Rbasis. This needs to correspond to the Bravais lattice
    cross_R2_R3 = jnp.cross(Rbasis[1], Rbasis[2])
    cross_R3_R1 = jnp.cross(Rbasis[2], Rbasis[0])
    cross_R1_R2 = jnp.cross(Rbasis[0], Rbasis[1])

    # Compute the volume (dot product of Rbasis[0] with cross_R2_R3)
    volume = jnp.dot(Rbasis[0], cross_R2_R3)

    # Compute b1, b2, b3 using the precomputed cross products and volume
    b1 = cross_R2_R3 / volume
    b2 = cross_R3_R1 / volume
    b3 = cross_R1_R2 / volume

    # Compute Gbasis matrix, scaled by 2*pi, each row is b1, b2, b3 scaled by 2pi
    Gbasis = 2 * jnp.pi * jnp.stack([b1, b2, b3])

    return Gbasis, volume


@jax.jit
def stress(Gbasis, dRbasis):
    # Call this for EVERY cell param
    two_pi = 2 * jnp.pi

    # Calculate db matrix using vectorized operations
    outer_products = jnp.einsum(
        "ik,jk->ij", Gbasis, dRbasis
    )  # Outer product: shape (3, 3)
    db = -jnp.matmul(outer_products, Gbasis) / two_pi  # Divide by 2pi

    # Calculate dGG matrix
    dGG = 2 * jnp.matmul(db, Gbasis.T)  # Shape (3, 3)

    return dGG


@jax.jit
def GsqStar_calc(kbz, Gbasis):
    # Where kbz is Nstars x 3 and Gbasis is 3 x 3
    Gbz = jnp.matmul(kbz, Gbasis)

    GsqStar = jnp.sum(Gbz**2, axis=1, keepdims=False)

    return GsqStar


@jax.jit
def Rbasis_calc(cellArray, angleArray):
    # cellArray is [L1, L2, L3], angleArray is [alpha, beta, gamma]
    # Calculate the Rbasis where each row is the Bravais lattice vector a1, a2, a3 and each column is x, y, z
    L1 = cellArray[0]
    L2 = cellArray[1]
    L3 = cellArray[2]
    alpha = angleArray[0]
    beta = angleArray[1]
    gamma = angleArray[2]

    f = (
        1
        - jnp.cos(alpha) ** 2
        - jnp.cos(beta) ** 2
        - jnp.cos(gamma) ** 2
        + 2 * jnp.cos(alpha)
        + jnp.cos(beta) * jnp.cos(gamma)
    )
    V = L1 * L2 * L3 * jnp.sqrt(f)

    Rbasis = jnp.array(
        [
            [L1, 0, 0],
            [L2 * jnp.cos(gamma), L2 * jnp.sin(gamma), 0],
            [
                L3 * jnp.cos(beta),
                L3 * (jnp.cos(alpha) - jnp.cos(beta) * jnp.cos(gamma)) / jnp.sin(gamma),
                V / (L1 * L2 * jnp.sin(gamma)),
            ],
        ]
    )

    return Rbasis


@jax.jit
def dRbasis_calc(cellArray, angleArray):
    # Calculate ALL possibilities of derivatives and use an indexing in the main code to choose which ones to investigate.
    L1 = cellArray[0]
    L2 = cellArray[1]
    L3 = cellArray[2]
    alpha = angleArray[0]
    beta = angleArray[1]
    gamma = angleArray[2]

    f = (
        1
        - jnp.cos(alpha) ** 2
        - jnp.cos(beta) ** 2
        - jnp.cos(gamma) ** 2
        + 2 * jnp.cos(alpha)
        + jnp.cos(beta) * jnp.cos(gamma)
    )
    V = L1 * L2 * L3 * jnp.sqrt(f)

    dV_alpha = (
        L1
        * L2
        * L3
        / jnp.sqrt(f)
        * jnp.sin(alpha)
        * (jnp.cos(alpha) - jnp.cos(beta) * jnp.cos(gamma))
    )
    dV_beta = (
        L1
        * L2
        * L3
        / jnp.sqrt(f)
        * jnp.sin(beta)
        * (jnp.cos(beta) - jnp.cos(alpha) * jnp.cos(gamma))
    )
    dV_gamma = (
        L1
        * L2
        * L3
        / jnp.sqrt(f)
        * jnp.sin(gamma)
        * (jnp.cos(gamma) - jnp.cos(alpha) * jnp.cos(beta))
    )

    # Derivatives of cell length parameters if all independent (L1 =/= L2 =/= L3)
    dR_0 = jnp.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])  # L1
    dR_1 = jnp.array([[0, 0, 0], [jnp.cos(gamma), jnp.sin(gamma), 0], [0, 0, 0]])  # L2
    dR_2 = jnp.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [
                jnp.cos(beta),
                (jnp.cos(alpha) - jnp.cos(beta) * jnp.cos(gamma)) / jnp.sin(gamma),
                V / (L1 * L2 * L3 * jnp.sin(gamma)),
            ],
        ]
    )  # L3

    # Derivatives of cell length parameters if coupled dependencies
    dR_3 = jnp.array(
        [
            [1, 0, 0],
            [jnp.cos(gamma), jnp.sin(gamma), 0],
            [
                jnp.cos(beta),
                (jnp.cos(alpha) - jnp.cos(beta) * jnp.cos(gamma)) / jnp.sin(gamma),
                V / (L1 * L2 * L3 * jnp.sin(gamma)),
            ],
        ]
    )  # L1 = L2 = L3
    dR_4 = jnp.array(
        [[1, 0, 0], [jnp.cos(gamma), jnp.sin(gamma), 0], [0, 0, 0]]
    )  # L1 = L2

    # Derivatives of angles if all independent
    dR_5 = jnp.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, -L3 * jnp.sin(alpha), dV_alpha / (L1 * L2 * jnp.sin(gamma))],
        ]
    )  # alpha
    dR_6 = jnp.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [
                -L3 * jnp.sin(beta),
                L3 * jnp.sin(beta) * jnp.cos(gamma),
                dV_beta / (L1 * L2 * jnp.sin(gamma)),
            ],
        ]
    )  # beta
    dR_7 = jnp.array(
        [
            [0, 0, 0],
            [-L2 * jnp.sin(gamma), L2 * jnp.cos(gamma), 0],
            [
                0,
                L3 * jnp.cos(beta) * jnp.sin(gamma),
                dV_gamma / (L1 * L2 * jnp.sin(gamma))
                - V / (L1 * L2 * jnp.sin(gamma) ** 2) * jnp.cos(gamma),
            ],
        ]
    )  # gamma

    # Derivatives of angles if alpha = beta = gamma (rhombohedral case)
    dR_8 = jnp.array(
        [
            [0, 0, 0],
            [-L2 * jnp.sin(gamma), L2 * jnp.cos(gamma), 0],
            [
                -L3 * jnp.sin(beta),
                L3 * jnp.sin(alpha) * (2 * jnp.cos(alpha) - 1),
                dV_alpha / (L1 * L2 * jnp.sin(gamma))
                - V / (L1 * L2 * L3 * jnp.sin(gamma) ** 2) * jnp.cos(alpha),
            ],
        ]
    )

    dR = jnp.array([dR_0, dR_1, dR_2, dR_3, dR_4, dR_5, dR_6, dR_7, dR_8])

    return dR

def calc_solvent_psi(omega_solv, FT):
    # Calculate the solvent psi from the solvent omega - NOT normalized to (1-c)
    # First, calculate the partition function of the solvent Q_solv, which is the space integral of exp(-omega_solv(x))
    u0_solv = jnp.exp(-omega_solv)  # Initial condition for solvent
    # Take the volume integral by transforming to star space and taking the first element
    u0_solv_star = jnp.matmul(FT, u0_solv)
    Q_solv = u0_solv_star[0]  # First element of star is a good approximate for integral
    psi_solv = jnp.exp(-omega_solv) / Q_solv  # Normalize to get psi_solv

    return psi_solv, Q_solv


# SCFT equation for use with gradient descent of cell and angle parameters
# For micelle simulations, includes the solvent chemical identity used to generate the solvent chi
# c is the fraction of the solution which is polymer
def scft(
    x, cell, angle, c, solvent, f_basis, space_group, u0_list, reg_param, dt, Ns, stencil, basis_fun
):
    # Extract the required parameters
    xi = x["xi"]
    psi = x["psi"]
    psi_solv = x['solvent_psi'] # This is the solvent psi - defined only spatially
    u0_forward, u0_backward = u0_list
    FT, FS, kbz, Gwt, dRIndex, cellIndex, angleIndex, angleArray = space_group

    alpha = reg_param["alpha"]
    beta = reg_param["beta"]
    gamma = reg_param["gamma"]
    target_energy = reg_param["target_energy"]

    # Mean shift xi because xi_star[0] = 0
    xi_star = jnp.matmul(FT, xi)
    xi_star = xi_star.at[0].set(0.0)
    xi = jnp.matmul(FS, xi_star)

    # Construct the function in s such that it is Ns x Np
    f_fun = jnp.dot(
        basis_fun.T, f_basis
    )  # basis_fun.T is Ns x Ncoeff, g is Ncoeff x Np

    # Calculate Lambda (interaction matrix)
    L_ss = pairwise(f_fun)
    L_expanded = L_ss[None, :, :]  # Dimensions of 1 x Ns x Ns
    psi_expanded = psi[:, :, None]  # Dimensions of Ntau x Ns x 1
    L_psi = (
        L_expanded * psi_expanded
    )  # Dimensions of Ntau x Ns x Ns which are tau x s' x s

    # Calculate for solvent
    # f_fun is a function of size Ns x Ndim. solvent must be Ndim. Take the squared difference for every row of f_fun, then take average in Ndim
    chi_solv = jnp.mean((f_fun - solvent) ** 2, axis=-1)  # chi is now Ns x 1
    chi_solv = chi_solv[None, :]  # Make chi 1 x Ns for broadcasting

    # Integrate over s' using stencil
    w_pred = jnp.einsum("ijk,jr->ik", L_psi, stencil) + (chi_solv * psi_solv) - xi

    # Get the w_solvent
    w_pred_solv = integral(chi_solv * psi, stencil) - xi

    # Rbasis = jnp.eye(3,3) # This the default Bravais lattice before modifications
    cellArray = jnp.ones((3,))  # Cell array of L1, L2, L3 - default is 1
    # angleArray # Cell array of angles alpha, beta, gamma - need to input because we sometimes have unregressed angles (hexagonal)

    for i in range(cell.shape[0]):
        # Assign to the cellArray
        cellArray = cellArray.at[cellIndex[i]].set(cell[i])

    for i in range(angle.shape[0]):
        # Assign to the angleArray
        angleArray = angleArray.at[angleIndex[i]].set(angle[i])

    # Construct Rbasis
    Rbasis = Rbasis_calc(cellArray, angleArray)
    # Construct Gbasis
    Gbasis, V = Gbasis_calc(Rbasis)
    # Construct GsqStar
    GsqStar = GsqStar_calc(kbz, Gbasis)
    # Construct dRbasis - changes if angle is one of the regressed parameters
    dRbasis_full = dRbasis_calc(cellArray, angleArray)
    dRbasis = dRbasis_full[
        dRIndex
    ]  # Directly access if dRIndex is jnp array and dRbasis_full is jnp array

    # Calculate the predicted psi via MDE
    # Note, use the solved psi for all future calculations
    psi_hat, Q_hat, int_u_prod_star = psi_solve(
        w_pred, u0_forward, u0_backward, dt, Ns, FT, FS, GsqStar, stencil
    )
    psi_hat = psi_hat * c

    # Calculate the predicted solvent psi via integration in x of exp(-ws(x) / N)
    # psi_solv_hat must be integrated over s
    psi_solv_hat, Q_solv_hat = calc_solvent_psi(w_pred_solv, FT)
    psi_solv_hat = psi_solv_hat * (1 - c)

    # Testing the SCFT equation constraints
    # We automatically satisfy dH/dw due to use of MDE, need to satisfy the other two
    # Incompressibility as a loss function term
    # We need to calculate the energy for not satisfying incompressibility
    incompress = integral(psi_hat, stencil) + psi_solv_hat - 1  # Integrate over s and subtract 1
    incompressible_MSE = jnp.mean((incompress) ** 2)
    incompressible_MAE = jnp.mean(jnp.absolute(incompress))
    size_incompress = jnp.size(incompress)
    incompressible_loss = incompressible_MSE
    incompress_report = incompressible_MAE

    # Proxy for the dH/dpsi equation, since we directly calculate w via psi, we can just ensure that the psi_hat we then calculate from that w is self-consistent
    psi_dif = psi - psi_hat
    psi_solv_dif = psi_solv - psi_solv_hat
    w_MSE = jnp.mean((psi_dif) ** 2) + jnp.mean((psi_solv_dif) ** 2)
    w_MAE = jnp.mean(jnp.absolute(psi_dif)) + jnp.mean(jnp.absolute(psi_solv_dif))
    size_w = jnp.size(psi_dif)
    w_loss = w_MSE
    w_report = w_MAE

    # Report the MATLAB criteria for the loss which is the 2-norm of the difference in star space
    psi_star = jnp.matmul(FT, psi)
    psi_hat_star = jnp.matmul(FT, psi_hat)
    w_matlab = jnp.sqrt(
        jnp.sum((psi_star[1:,:] - psi_hat_star[1:,:]) ** 2) / jnp.sum(psi_hat_star[1:,:]**2)
    )

    # Don't include the deviation from incompressibility as an energy term
    # Use psi_hat (the calculated field) instead of psi to avoid driving to homogeneous field
    F = hamiltonian(c, Q_hat, Q_solv_hat, psi_hat, psi_solv_hat, w_pred, w_pred_solv, L_ss, chi_solv, dt, FT, FS)  # - incompress_energy[0]

    # Disorder energy loss term
    F_disorder = hamiltonian(c, 
        1, 1, c * jnp.ones_like(psi_hat), (1-c) * jnp.ones_like(psi_solv_hat), jnp.zeros_like(w_pred), jnp.zeros_like(w_pred_solv), L_ss, chi_solv, dt, FT, FS
    )

    # Flatten both energies
    F = F.flatten()[0] if F.ndim > 0 else F
    F_disorder = F_disorder.flatten()[0] if F_disorder.ndim > 0 else F_disorder

    # Calculate the stress - for reporting purposes only
    cell_stress = jnp.zeros(dRbasis.shape[0])

    # Do the calculation for each cell
    for i in range(dRbasis.shape[0]):
        dGG = stress(Gbasis, dRbasis[i])  # 3 x 3 matrix

        # Calculate vectorized for every star that we have. Result should be a single vector Rs of size Nstars
        dGsq = jnp.einsum("im,iv,mv->i", kbz, kbz, dGG)  # Result shape: (Ns,)

        Rs = c * int_u_prod_star.flatten() * dGsq * Gwt**2
        #Rs_solv = (1 - c) * int_u_prod_star_solv.flatten() * dGsq * Gwt**2

        stress_d = 0.0 # Turn off stress for micelle #1 / 6 * 1 / Q_hat * 1 / V * jnp.sum(Rs)

        cell_stress = cell_stress.at[i].set(stress_d)

    # stress_loss = jnp.mean((cell_stress)**2) # Calculated but unused

    # Loss function
    # We want to minimize free energy to avoid the disordered phase. Use the calculated values
    scft_loss = (
        w_loss + incompressible_loss
    )  # Satisfaction of SCFT constraints - equal weighting not desired
    # print sizes of each for troubleshooting
    # scft_loss = (size_w * w_loss + size_incompress * incompressible_loss) / (size_w + size_incompress) # Satisfaction of SCFT constraints - equal weighting
    free_energy_loss = jnp.absolute(F)  # Minimization of structure free energy
    energy_dif_loss = jax.nn.softplus(
        F - target_energy
    )  # Penalization if the regressed structure has greater free energy than the target

    loss = alpha * scft_loss + beta * free_energy_loss + gamma * energy_dif_loss

    return (loss, (w_matlab, w_report, incompress_report, cell_stress, F, F_disorder))


# scftRefine shifts the psi fields to always enforce incompressibility, good for refining a structure when the supplied guess is near a converged solution
# We only have the scft loss term
def scftRefine(
    x, cell, angle, c, solvent, f_basis, space_group, u0_list, dt, Ns, stencil, basis_fun
):
    # Extract the required parameters
    xi = x["xi"]
    psi = x["psi"]
    psi_solv = x['solvent_psi'] # This is the solvent psi - defined only spatially
    xi_unmod = xi.copy()
    psi_unmod = psi.copy()  # This is to penalize unconvergent solutions
    psi_solv_unmod = psi_solv.copy()
    u0_forward, u0_backward = u0_list
    FT, FS, kbz, Gwt, dRIndex, cellIndex, angleIndex, angleArray = space_group

    # Mean shift xi because xi_star[0] = 0
    xi_star = jnp.matmul(FT, xi)
    xi_star = xi_star.at[0].set(0.0)
    xi = jnp.matmul(FS, xi_star)

    # Mean shift psi so that average is 1 - only do in the refinement step - do not do this, naturally let this happen with mean shift of psi_hat
    #psi_star = jnp.matmul(FT, psi)
    #psi_star = psi_star.at[0, :].set(1.0)
    #psi = jnp.matmul(FS, psi_star)  # Convert back to tau space

    # Construct the function in s such that it is Ns x Np
    f_fun = jnp.dot(
        basis_fun.T, f_basis
    )  # basis_fun.T is Ns x Ncoeff, g is Ncoeff x Np

    # Calculate Lambda (interaction matrix)
    L_ss = pairwise(f_fun)
    L_expanded = L_ss[None, :, :]  # Dimensions of 1 x Ns x Ns
    psi_expanded = psi[:, :, None]  # Dimensions of Ntau x Ns x 1
    L_psi = (
        L_expanded * psi_expanded
    )  # Dimensions of Ntau x Ns x Ns which are tau x s' x s

    # Calculate for solvent
    # f_fun is a function of size Ns x Ndim. solvent must be Ndim. Take the squared difference for every row of f_fun, then take average in Ndim
    chi_solv = jnp.mean((f_fun - solvent) ** 2, axis=-1)  # chi is now Ns x 1
    chi_solv = chi_solv[None, :]  # Make chi 1 x Ns for broadcasting

    # Integrate over s' using stencil
    w_pred = jnp.einsum("ijk,jr->ik", L_psi, stencil) + (chi_solv * psi_solv) - xi

    # Get the w_solvent
    w_pred_solv = integral(chi_solv * psi, stencil) - xi

    # Rbasis = jnp.eye(3,3) # This the default Bravais lattice before modifications
    cellArray = jnp.ones((3,))  # Cell array of L1, L2, L3 - default is 1

    for i in range(cell.shape[0]):
        # Assign to the cellArray
        cellArray = cellArray.at[cellIndex[i]].set(cell[i])

    for i in range(angle.shape[0]):
        # Assign to the angleArray
        angleArray = angleArray.at[angleIndex[i]].set(angle[i])

    # Construct Rbasis
    Rbasis = Rbasis_calc(cellArray, angleArray)
    # Construct Gbasis
    Gbasis, V = Gbasis_calc(Rbasis)
    # Construct GsqStar
    GsqStar = GsqStar_calc(kbz, Gbasis)
    # Construct dRbasis - changes if angle is one of the regressed parameters
    dRbasis_full = dRbasis_calc(cellArray, angleArray)
    dRbasis = dRbasis_full[
        dRIndex
    ]  # Directly access if dRIndex is jnp array and dRbasis_full is jnp array

    # Calculate the predicted psi via MDE
    # Note, use the solved psi for all future calculations
    psi_hat, Q_hat, int_u_prod_star = psi_solve(
        w_pred, u0_forward, u0_backward, dt, Ns, FT, FS, GsqStar, stencil
    )
    psi_hat = psi_hat * c

    # Calculate the predicted solvent psi via integration in x of exp(-ws(x) / N)
    # psi_solv_hat must be integrated over s
    psi_solv_hat, Q_solv_hat = calc_solvent_psi(w_pred_solv, FT)
    psi_solv_hat = psi_solv_hat * (1 - c)

    # Testing the SCFT equation constraints
    # We automatically satisfy dH/dw due to use of MDE, need to satisfy the other two
    # Incompressibility as a loss function term
    # We need to calculate the energy for not satisfying incompressibility
    psi_hat_star = jnp.matmul(FT, psi_hat)

    # Regularize the zeroth wavevector to have the correct average density
    psi_hat_star = psi_hat_star.at[0, :].set(c)
    psi_hat = jnp.matmul(FS, psi_hat_star)  # Convert back to tau space

    psi_solv_hat_star = jnp.matmul(FT, psi_solv_hat)
    psi_solv_hat_star = psi_solv_hat_star.at[0, :].set(1 - c)
    psi_solv_hat = jnp.matmul(FS, psi_solv_hat_star)  # Convert back to tau space
    
    incompress = integral(psi_hat, stencil) + psi_solv_hat - 1  # Integrate over s and subtract 1

    incompressible_MSE = jnp.mean((incompress) ** 2)
    incompressible_MAE = jnp.mean(jnp.absolute(incompress))
    size_incompress = jnp.size(incompress)
    # Report MAE, use MSE for optimization
    incompressible_loss = incompressible_MSE
    incompress_report = incompressible_MAE

    # Proxy for the dH/dpsi equation, since we directly calculate w via psi, we can just ensure that the psi_hat we then calculate from that w is self-consistent
    psi_dif = psi - psi_hat
    psi_solv_dif = psi_solv - psi_solv_hat

    # Don't take difference in zeroth wavevector, since this is changed separately outside the loop
    w_MSE = jnp.mean((psi_dif) ** 2) + jnp.mean((psi_solv_dif) ** 2)
    w_MAE = jnp.mean(jnp.absolute(psi_dif))
    size_w = jnp.size(psi_dif)
    # Report MAE, use MSE for optimization
    w_loss = w_MSE
    w_report = w_MAE

    # Report the MATLAB criteria for the loss which is the 2-norm of the difference in star space
    psi_star = jnp.matmul(FT, psi)
    psi_hat_star = jnp.matmul(FT, psi_hat)
    w_matlab = jnp.sqrt(
        jnp.sum((psi_star[1:,:] - psi_hat_star[1:,:]) ** 2) / jnp.sum(psi_hat_star[1:,:]**2)
    )

    # Don't include the deviation from incompressibility as an energy term
    # Use psi_hat (the calculated field) instead of psi to avoid driving to homogeneous field
    F = hamiltonian(c, Q_hat, Q_solv_hat, psi_hat, psi_solv_hat, w_pred, w_pred_solv, L_ss, chi_solv, dt, FT, FS)  # - incompress_energy[0]

    # Disorder energy loss term
    F_disorder = hamiltonian(c, 
        1, 1, c * jnp.ones_like(psi_hat), (1-c) * jnp.ones_like(psi_solv_hat), jnp.zeros_like(w_pred), jnp.zeros_like(w_pred_solv), L_ss, chi_solv, dt, FT, FS
    )

    # Flatten both energies
    F = F.flatten()[0] if F.ndim > 0 else F
    F_disorder = F_disorder.flatten()[0] if F_disorder.ndim > 0 else F_disorder

    # Calculate the stress
    cell_stress = jnp.zeros(dRbasis.shape[0])

    # Do the calculation for each cell
    for i in range(dRbasis.shape[0]):
        dGG = stress(Gbasis, dRbasis[i])  # 3 x 3 matrix

        # Calculate vectorized for every star that we have. Result should be a single vector Rs of size Nstars
        dGsq = jnp.einsum("im,iv,mv->i", kbz, kbz, dGG)  # Result shape: (Ns,)

        Rs = c * int_u_prod_star.flatten() * dGsq * Gwt**2
        #Rs_solv = (1 - c) * int_u_prod_star_solv.flatten() * dGsq * Gwt**2

        stress_d = 0.0 # Turn off stress # 1 / 6 * 1 / Q_hat * 1 / V * jnp.sum(Rs)

        cell_stress = cell_stress.at[i].set(stress_d)

    # Loss function - this gives significant weight to the incompressible loss when converging large 3D structures
    # We can weight this to give equal weight to both
    # scft_loss = (size_w * w_loss + size_incompress * incompressible_loss) / (size_w + size_incompress) # Satisfaction of SCFT constraints - equal weighting

    psi_loss = jnp.mean((psi_unmod - psi) ** 2)
    xi_loss = jnp.mean((xi_unmod - xi) ** 2)
    psi_solv_loss = jnp.mean((psi_solv_unmod - psi_solv) ** 2)

    scft_loss = (
        w_loss + incompressible_loss + psi_loss + xi_loss + psi_solv_loss
    )  # Equal weighting not desired

    loss = scft_loss

    return (loss, (w_matlab, w_report, incompress_report, cell_stress, F, F_disorder))


# Use only for the initial inverse design against a target. This allows the cell parameter to be changed at the same time, since the initial guess is very far from any known solutions
def scftID(x, c, phi_target, space_group, u0_list, reg_param, dt, Ns, stencil, basis_fun):
    # Unpack the regressible variables for the target structure
    f_basis = x["f_basis"]
    solvent = x["solvent"]
    xi = x["xi"]
    psi = x["psi"]
    psi_solv = x['solvent_psi'] # This is the solvent psi - defined only spatially
    cell = x["cell"]
    angle = (
        x["angle"] if "angle" in x else jnp.array([])
    )  # Assign empty list if no regressible angles

    u0_forward, u0_backward = u0_list
    FT, FS, kbz, Gwt, dRIndex, cellIndex, angleIndex, angleArray = space_group

    # Unpack regularization parameters
    alpha = reg_param["phi"]
    beta = reg_param["scft"]
    gamma = reg_param["energy"]
    delta = reg_param["chebyshev"]

    # Mean shift xi because xi_star[0] = 0 # Xi can be freely shifted
    xi_unmod = xi.copy()
    xi_star = jnp.matmul(FT, xi)
    xi_star = xi_star.at[0].set(0.0)
    xi = jnp.matmul(FS, xi_star)

    # Construct the function in s such that it is Ns x Np
    f_fun = jnp.dot(
        basis_fun.T, f_basis
    )  # basis_fun.T is Ns x Ncoeff, g is Ncoeff x Np

    # Calculate Lambda (interaction matrix)
    L_ss = pairwise(f_fun)
    L_expanded = L_ss[None, :, :]  # Dimensions of 1 x Ns x Ns
    psi_expanded = psi[:, :, None]  # Dimensions of Ntau x Ns x 1
    L_psi = (
        L_expanded * psi_expanded
    )  # Dimensions of Ntau x Ns x Ns which are tau x s' x s

    # Calculate for solvent
    # f_fun is a function of size Ns x Ndim. solvent must be Ndim. Take the squared difference for every row of f_fun, then take average in Ndim
    chi_solv = jnp.mean((f_fun - solvent) ** 2, axis=-1)  # chi is now Ns x 1
    chi_solv = chi_solv[None, :]  # Make chi 1 x Ns for broadcasting

    # Integrate over s' using stencil
    w_pred = jnp.einsum("ijk,jr->ik", L_psi, stencil) + (chi_solv * psi_solv) - xi

    # Get the w_solvent
    w_pred_solv = integral(chi_solv * psi, stencil) - xi

    # Rbasis = jnp.eye(3,3) # This the default Bravais lattice before modifications
    cellArray = jnp.ones((3,))  # Cell array of L1, L2, L3 - default is 1
    # angleArray # Cell array of angles alpha, beta, gamma - need to input because we sometimes have unregressed angles (hexagonal)

    for i in range(cell.shape[0]):
        # Assign to the cellArray
        cellArray = cellArray.at[cellIndex[i]].set(cell[i])

    for i in range(angle.shape[0]):
        # Assign to the angleArray
        angleArray = angleArray.at[angleIndex[i]].set(angle[i])

    # Construct Rbasis
    Rbasis = Rbasis_calc(cellArray, angleArray)
    # Construct Gbasis
    Gbasis, V = Gbasis_calc(Rbasis)
    # Construct GsqStar
    GsqStar = GsqStar_calc(kbz, Gbasis)
    # Construct dRbasis - changes if angle is one of the regressed parameters
    dRbasis_full = dRbasis_calc(cellArray, angleArray)
    dRbasis = dRbasis_full[
        dRIndex
    ]  # Directly access if dRIndex is jnp array and dRbasis_full is jnp array

    # Calculate the predicted psi via MDE
    # Note, use the solved psi for all future calculations
    psi_hat, Q_hat, int_u_prod_star = psi_solve(
        w_pred, u0_forward, u0_backward, dt, Ns, FT, FS, GsqStar, stencil
    )
    psi_hat = psi_hat * c

    # Calculate the predicted solvent psi via integration in x of exp(-ws(x) / N)
    # psi_solv_hat must be integrated over s
    psi_solv_hat, Q_solv_hat = calc_solvent_psi(w_pred_solv, FT)
    psi_solv_hat = psi_solv_hat * (1 - c)

    # Calculate the chemical identity field phi
    # sequence_to_identity takes (fun, dimension, crossover, width - >1 narrow, <1 broad)
    g_fun = sequence_to_identity(f_fun)  # This is (Ns, )
    phi_hat = integral(psi_hat * g_fun, stencil) # This is (Ntaus, 1)
    #phi = integral(psi * g_fun, stencil)

    # Calculate the difference between the target and prediction
    # What if we do the deviation in star space?
    phi_dev_loss = jnp.mean(
        (phi_target - phi_hat) ** 2
    )
    phi_report = jnp.mean(jnp.absolute(phi_target - phi_hat))

    # Incompressibility as a loss function term
    incompress = integral(psi_hat, stencil) + psi_solv_hat - 1  # Integrate over s and subtract 1
    # incompress_xi = jnp.matmul(FT, xi * incompress) # Multiply xi and the incompressibilty condition, transform to star space for integration
    # incompress_energy = incompress_xi[0]

    # Incompressibility loss term
    incompressible_MSE = jnp.mean((incompress) ** 2)
    incompressible_MAE = jnp.mean(jnp.absolute(incompress))
    size_incompress = jnp.size(incompress)
    incompressible_loss = incompressible_MSE
    incompress_report = incompressible_MAE

    # Satisfaction of the SCFT equation is required, since no longer strictly enforced, we penalize if the learned density field is different from the guessed density field
    psi_dif = psi - psi_hat
    psi_solv_dif = psi_solv - psi_solv_hat

    w_MSE = jnp.mean((psi_dif) ** 2) + jnp.mean((psi_solv_dif) ** 2)
    w_MAE = jnp.mean(jnp.absolute(psi_dif))
    size_w = jnp.size(psi_dif)
    w_loss = w_MSE
    w_report = w_MAE

    # Report the MATLAB criteria for the loss which is the 2-norm of the difference in star space
    psi_star = jnp.matmul(FT, psi)
    psi_hat_star = jnp.matmul(FT, psi_hat)
    w_matlab = jnp.sqrt(
        jnp.sum((psi_star[1:,:] - psi_hat_star[1:,:]) ** 2) / jnp.sum(psi_hat_star[1:,:]**2)
    )

    # Calculate the stress
    cell_stress = jnp.zeros(dRbasis.shape[0])
    size_stress = jnp.size(cell_stress)

    # Do the calculation for each cell
    for i in range(dRbasis.shape[0]):
        dGG = stress(Gbasis, dRbasis[i])  # 3 x 3 matrix

        # Calculate vectorized for every star that we have. Result should be a single vector Rs of size Nstars
        dGsq = jnp.einsum("im,iv,mv->i", kbz, kbz, dGG)  # Result shape: (Ns,)

        Rs = c * int_u_prod_star.flatten() * dGsq * Gwt**2
        #Rs_solv = (1 - c) * int_u_prod_star_solv.flatten() * dGsq * Gwt**2

        stress_d = 0.0 #  Turn off stress # 1 / 6 * 1 / Q_hat * 1 / V * jnp.sum(Rs)

        cell_stress = cell_stress.at[i].set(stress_d)

    stress_loss = jnp.mean((cell_stress) ** 2)

    # Energy of current structure accounting for not satisfying incompressibility
    # Don't need incompressible energy, use the calculated density field structure
    F = hamiltonian(c, Q_hat, Q_solv_hat, psi_hat, psi_solv_hat, w_pred, w_pred_solv, L_ss, chi_solv, dt, FT, FS)  # - incompress_energy[0]

    # Disorder energy loss term
    F_disorder = hamiltonian(c, 
        1, 1, c * jnp.ones_like(psi_hat), (1-c) * jnp.ones_like(psi_solv_hat), jnp.zeros_like(w_pred), jnp.zeros_like(w_pred_solv), L_ss, chi_solv, dt, FT, FS
    )

    # Flatten both energies
    F = F.flatten()[0] if F.ndim > 0 else F
    F_disorder = F_disorder.flatten()[0] if F_disorder.ndim > 0 else F_disorder

    # Loss terms
    xi_loss = jnp.mean((xi_unmod - xi) ** 2)
    scft_loss = (
        w_loss + incompressible_loss + stress_loss + xi_loss
    )  # Equal weighting not desired
    energy_loss = F # jnp.relu(F - F_disorder - 0.01)  # Penalize disorder here because we don't have negative selection
    cheby_reg = jnp.mean(f_basis**2)  # Regularization term - L2 (?) norm

    loss = (
        alpha * phi_dev_loss
        + beta * scft_loss
        + gamma * energy_loss
        + delta * cheby_reg
    )

    return (
        loss,
        (
            w_matlab,
            w_report,
            incompress_report,
            cell_stress,
            F,
            F_disorder,
            phi_hat,
            phi_report,
        ),
    )

@jax.jit
def evaluateEnergyInverse(x, c, space_group, u0_list, dt, Ns, stencil, basis_fun):
    # NO mean shifting of input psi fields. Ok to mean shift the calculated psi_hat because in theory the scftRefine step should have converged psi and psi_hat 
    # Unpack the regressible variables for the target structure
    f_basis = x["f_basis"]
    solvent = x["solvent"]
    xi = x["xi"]
    psi = x["psi"]
    psi_solv = x['solvent_psi'] # This is the solvent psi - defined only spatially
    cell = x["cell"]
    angle = (
        x["angle"] if "angle" in x else jnp.array([])
    )  # Assign empty list if no regressible angles
    u0_forward, u0_backward = u0_list
    FT, FS, kbz, Gwt, dRIndex, cellIndex, angleIndex, angleArray = space_group

    # Mean shift xi because xi_star[0] = 0 # Xi can be freely shifted
    xi_star = jnp.matmul(FT, xi)
    xi_star = xi_star.at[0].set(0.0)
    xi = jnp.matmul(FS, xi_star)

    # Construct the function in s such that it is Ns x Np
    f_fun = jnp.dot(
        basis_fun.T, f_basis
    )  # basis_fun.T is Ns x Ncoeff, g is Ncoeff x Np

    # f_fun is a function of size Ns x Ndim. solvent must be Ndim. Take the squared difference for every row of f_fun, then take average in Ndim
    chi_solv = jnp.mean((f_fun - solvent) ** 2, axis=-1)  # chi is now Ns x 1
    chi_solv = chi_solv[None, :]  # Make chi 1 x Ns for broadcasting

    # Calculate Lambda (interaction matrix)
    L_ss = pairwise(f_fun)
    L_expanded = L_ss[None, :, :]  # Dimensions of 1 x Ns x Ns
    psi_expanded = psi[:, :, None]  # Dimensions of Ntau x Ns x 1
    L_psi = (
        L_expanded * psi_expanded
    )  # Dimensions of Ntau x Ns x Ns which are tau x s' x s

    # Integrate over s' using stencil
    w_pred = jnp.einsum("ijk,jr->ik", L_psi, stencil) + (chi_solv * psi_solv) - xi

    # Get the w_solvent
    w_pred_solv = integral(chi_solv * psi, stencil) - xi

    # Rbasis = jnp.eye(3,3) # This the default Bravais lattice before modifications
    cellArray = jnp.ones((3,))  # Cell array of L1, L2, L3 - default is 1
    # angleArray # Cell array of angles alpha, beta, gamma - need to input because we sometimes have unregressed angles (hexagonal)

    for i in range(cell.shape[0]):
        # Assign to the cellArray
        cellArray = cellArray.at[cellIndex[i]].set(cell[i])

    for i in range(angle.shape[0]):
        # Assign to the angleArray
        angleArray = angleArray.at[angleIndex[i]].set(angle[i])

    # Construct Rbasis
    Rbasis = Rbasis_calc(cellArray, angleArray)
    # Construct Gbasis
    Gbasis, V = Gbasis_calc(Rbasis)
    # Construct GsqStar
    GsqStar = GsqStar_calc(kbz, Gbasis)
    # Construct dRbasis - changes if angle is one of the regressed parameters
    dRbasis_full = dRbasis_calc(cellArray, angleArray)
    dRbasis = dRbasis_full[
        dRIndex
    ]  # Directly access if dRIndex is jnp array and dRbasis_full is jnp array

    # Calculate the predicted psi via MDE
    # Note, use the solved psi for all future calculations
    psi_hat, Q_hat, _ = psi_solve(
        w_pred, u0_forward, u0_backward, dt, Ns, FT, FS, GsqStar, stencil
    )

    # Normalize psi_hat - This is to correct any error from the MDE. This is allowable since after scftRefine, the psi - psi* should be very low, and psi was normalized in SCFT refine
    psi_hat_star = jnp.matmul(FT, psi_hat)
    psi_hat_star = psi_hat_star.at[0, :].set(c)
    psi_hat = jnp.matmul(FS, psi_hat_star)  # Convert back to tau space

    # Calculate the predicted solvent psi via integration in x of exp(-ws(x) / N)
    # psi_solv_hat must be integrated over s
    psi_solv_hat, Q_solv_hat = calc_solvent_psi(w_pred_solv, FT)
    psi_solv_hat = psi_solv_hat * (1 - c)

    # Energy of current structure accounting for not satisfying incompressibility
    # Don't need incompressible energy, use the calculated density field structure
    F = hamiltonian(c, Q_hat, Q_solv_hat, psi_hat, psi_solv_hat, w_pred, w_pred_solv, L_ss, chi_solv, dt, FT, FS)

    # Disorder energy loss term
    F_disorder = hamiltonian(c, 
        1, 1, c * jnp.ones_like(psi_hat), (1-c) * jnp.ones_like(psi_solv_hat), jnp.zeros_like(w_pred), jnp.zeros_like(w_pred_solv), L_ss, chi_solv, dt, FT, FS
    )

    # Flatten both energies
    F = F.flatten()[0] if F.ndim > 0 else F
    F_disorder = F_disorder.flatten()[0] if F_disorder.ndim > 0 else F_disorder

    return F, F_disorder


@jax.jit
def evaluateFields(x, c, space_group, u0_list, dt, Ns, stencil, basis_fun):
    # Return values in the SCFT equations necessary for reproduction in other software
    # Unpack the regressible variables for the target structure
    f_basis = x["f_basis"]
    solvent = x["solvent"]
    xi = x["xi"]
    psi = x["psi"]
    psi_solv = x['solvent_psi'] # This is the solvent psi - defined only spatially
    cell = x["cell"]
    angle = (
        x["angle"] if "angle" in x else jnp.array([])
    )  # Assign empty list if no regressible angles
    u0_forward, u0_backward = u0_list
    FT, FS, kbz, Gwt, dRIndex, cellIndex, angleIndex, angleArray = space_group

    # Mean shift xi because xi_star[0] = 0
    xi_star = jnp.matmul(FT, xi)
    xi_star = xi_star.at[0].set(0.0)
    xi = jnp.matmul(FS, xi_star)

    # Normalize psi
    #psi_star = jnp.matmul(FT, psi)
    #psi_star = psi_star.at[0, :].set(1.0)
    #psi = jnp.matmul(FS, psi_star)  # Convert back to tau space

    # Construct the function in s such that it is Ns x Np
    f_fun = jnp.dot(
        basis_fun.T, f_basis
    )  # basis_fun.T is Ns x Ncoeff, g is Ncoeff x Np

    # Calculate Lambda (interaction matrix)
    L_ss = pairwise(f_fun)
    L_expanded = L_ss[None, :, :]  # Dimensions of 1 x Ns x Ns
    psi_expanded = psi[:, :, None]  # Dimensions of Ntau x Ns x 1
    L_psi = (
        L_expanded * psi_expanded
    )  # Dimensions of Ntau x Ns x Ns which are tau x s' x s

    # f_fun is a function of size Ns x Ndim. solvent must be Ndim. Take the squared difference for every row of f_fun, then take average in Ndim
    chi_solv = jnp.mean((f_fun - solvent) ** 2, axis=-1)  # chi is now Ns x 1
    chi_solv = chi_solv[None, :]  # Make chi 1 x Ns for broadcasting

    # Integrate over s' using stencil
    w_pred = jnp.einsum("ijk,jr->ik", L_psi, stencil) + (chi_solv * psi_solv) - xi

    # Get the w_solvent
    w_pred_solv = integral(chi_solv * psi, stencil) - xi

    # Rbasis = jnp.eye(3,3) # This the default Bravais lattice before modifications
    cellArray = jnp.ones((3,))  # Cell array of L1, L2, L3 - default is 1
    # angleArray # Cell array of angles alpha, beta, gamma - need to input because we sometimes have unregressed angles (hexagonal)

    for i in range(cell.shape[0]):
        # Assign to the cellArray
        cellArray = cellArray.at[cellIndex[i]].set(cell[i])

    for i in range(angle.shape[0]):
        # Assign to the angleArray
        angleArray = angleArray.at[angleIndex[i]].set(angle[i])

    # Construct Rbasis
    Rbasis = Rbasis_calc(cellArray, angleArray)
    # Construct Gbasis
    Gbasis, V = Gbasis_calc(Rbasis)
    # Construct GsqStar
    GsqStar = GsqStar_calc(kbz, Gbasis)
    # Construct dRbasis - changes if angle is one of the regressed parameters
    dRbasis_full = dRbasis_calc(cellArray, angleArray)
    dRbasis = dRbasis_full[
        dRIndex
    ]  # Directly access if dRIndex is jnp array and dRbasis_full is jnp array

    # Calculate the predicted psi via MDE
    # Note, use the solved psi for all future calculations
    psi_hat, Q_hat, int_u_prod_star = psi_solve(
        w_pred, u0_forward, u0_backward, dt, Ns, FT, FS, GsqStar, stencil
    )
    psi_hat = psi_hat * c

    # Normalization
    psi_hat_star = jnp.matmul(FT, psi_hat)
    psi_hat_star = psi_hat_star.at[0, :].set(c)
    psi_hat = jnp.matmul(FS, psi_hat_star)  # Convert back to tau space

    # Calculate the predicted solvent psi via integration in x of exp(-ws(x) / N)
    # psi_solv_hat must be integrated over s
    psi_solv_hat, Q_solv_hat = calc_solvent_psi(w_pred_solv, FT)
    psi_solv_hat = psi_solv_hat * (1 - c)

    # Normalization
    psi_solv_hat_star = jnp.matmul(FT, psi_solv_hat)
    psi_solv_hat_star = psi_solv_hat_star.at[0].set(1 - c)
    psi_solv_hat = jnp.matmul(FS, psi_solv_hat_star)  # Convert back to tau space

    # Calculate the chemical identity field phi
    # For testing, we are hard-coding what we consider a transition
    # sequence_to_identity takes (fun, dimension, crossover, width - >1 narrow, <1 broad)
    g_fun = sequence_to_identity(f_fun)  # This is (Ns, )
    phi_hat = integral(psi_hat * g_fun, stencil)  # This is (Ntaus, 1

    # Incompressibility as a loss function term
    incompress = integral(psi_hat, stencil) + psi_solv_hat - 1  # Integrate over s and subtract 1
    # incompress_xi = jnp.matmul(FT, xi * incompress) # Multiply xi and the incompressibilty condition, transform to star space for integration
    # incompress_energy = incompress_xi[0]

    # Calculate the stress
    cell_stress = jnp.zeros(dRbasis.shape[0])

    # Do the calculation for each cell
    for i in range(dRbasis.shape[0]):
        dGG = stress(Gbasis, dRbasis[i])  # 3 x 3 matrix

        # Calculate vectorized for every star that we have. Result should be a single vector Rs of size Nstars
        dGsq = jnp.einsum("im,iv,mv->i", kbz, kbz, dGG)  # Result shape: (Ns,)

        Rs = c * int_u_prod_star.flatten() * dGsq * Gwt**2
        #Rs_solv = (1 - c) * int_u_prod_star_solv.flatten() * dGsq * Gwt**2

        stress_d = 0.0 # Turn off stress # 1 / 6 * 1 / Q_hat * 1 / V * jnp.sum(Rs)

        cell_stress = cell_stress.at[i].set(stress_d)

    # Energy of current structure accounting for not satisfying incompressibility
    # Don't need incompressible energy, use the calculated density field structure
    F = hamiltonian(c, Q_hat, Q_solv_hat, psi_hat, psi_solv_hat, w_pred, w_pred_solv, L_ss, chi_solv, dt, FT, FS)  # - incompress_energy[0]

    # Disorder energy loss term
    F_disorder = hamiltonian(c, 
        1, 1, c * jnp.ones_like(psi_hat), (1-c) * jnp.ones_like(psi_solv_hat), jnp.zeros_like(w_pred), jnp.zeros_like(w_pred_solv), L_ss, chi_solv, dt, FT, FS
    )

    # Flatten both energies
    F = F.flatten()[0] if F.ndim > 0 else F
    F_disorder = F_disorder.flatten()[0] if F_disorder.ndim > 0 else F_disorder

    # Report the MATLAB criteria for the loss which is the 2-norm of the difference in star space
    psi_star = jnp.matmul(FT, psi)
    psi_hat_star = jnp.matmul(FT, psi_hat)
    w_matlab = jnp.sqrt(
        jnp.sum((psi_star[1:,:] - psi_hat_star[1:,:]) ** 2) / jnp.sum(psi_hat_star[1:,:]**2)
    )

    output_dict = {
        "phi": phi_hat,
        "psi_scft": psi_hat,
        "w": w_pred,
        "w_matlab": w_matlab,
        "incompressibility": incompress,
        "stress": cell_stress,
        "energy": F,
        "disorder energy": F_disorder,
        "Q": Q_hat,
        "L_ss": L_ss,
        "chi_solv": chi_solv,
        "psi_solv": psi_solv_hat,
        "w_solv": w_pred_solv,
        "Q_solv": Q_solv_hat
    }

    return output_dict
