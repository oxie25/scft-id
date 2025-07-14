# Additional helper functions for inverse design & negative selection
import jax.numpy as jnp
import jax
import optax
import equinox as eqx
from functools import partial

from .scft_utils import (
    scft,
    scftRefine,
    evaluateEnergyInverse,
    sequence_to_identity,
    integral,
)


class IDClass(eqx.Module):
    dt: float
    Ns: int
    stencil: jnp.ndarray
    basis: jnp.ndarray
    space_group: dict = eqx.static_field()
    transform_fun: callable = eqx.static_field()
    sg_alt_dict: dict = eqx.static_field()


# Calculate the probabilities for each structure
def calculate_probabilities(final_energy_array, F, sharpness=1e2):
    # Calculate the modified probability distribution
    energy_differences = (final_energy_array - F) / F  # MPE
    boltzmann_weights = jnp.where(
        energy_differences <= 0,
        jnp.exp(
            -sharpness * energy_differences
        ),  # Enhance probability for negative differences
        jnp.exp(
            -sharpness * energy_differences
        ),  # Sharpen decay for positive differences
    )
    alt_probability = boltzmann_weights / jnp.sum(boltzmann_weights)
    return alt_probability


# Calculate the loss term for the energy comparison
def energy_loss_fn(target_energy, alternate_energy, offset):
    # Alternate energy is either jax array or float
    # Default is take ratio
    loss = jnp.mean(
        jax.nn.relu(
            jnp.absolute(target_energy) / jnp.absolute(alternate_energy) - 1 / offset
        )
    )

    # We can also take the difference
    # loss = jnp.mean(jax.nn.relu(target_energy - alternate_energy + offset))

    # Take the softplus of the difference, this ensures there's a penalty for approaching the disorder region
    # loss = jnp.mean(jax.nn.softplus(offset * (target_energy - alternate_energy)))

    return loss


# This turns the refining loop into a jitted function
# The input variables should not have gradients tracked
@partial(jax.jit, static_argnames=["refine_epoch"])
def scft_refining_inner_loop(
    id_class, x_refine, f_basis, sg, u0, lr, cell_lr, refine_epoch
):
    # We want to jax.jit the entire refining function, with fixed steps (no early stopping)
    x_opt = {"xi": x_refine["xi"], "psi": x_refine["psi"]}
    cell = x_refine["cell"]
    angle = x_refine["angle"] if "angle" in x_refine else jnp.array([])
    # Set up the optimizer
    optimizer_energy = optax.adam(learning_rate=0.01, nesterov=True)
    optimizer_refine = optax.adam(learning_rate=lr, nesterov=True)
    opt_state_refine = optimizer_refine.init(x_opt)
    f_param = id_class.transform_fun(
        f_basis
    )  # id_class.sequence.transform_fun(f_basis)
    reg_param = {"alpha": 1, "beta": 0, "gamma": 1, "target_energy": 0}

    def step_energy(carry, _):
        opt_state, x, cell, angle = carry
        # compute loss + grads
        # the scft here comes from scft_utils.scft
        (loss_value, aux), grads = jax.value_and_grad(scft, has_aux=True)(
            x,
            cell,
            angle,
            f_param,
            sg,
            u0,
            reg_param,
            id_class.dt,
            id_class.Ns,
            id_class.stencil,
            id_class.basis,
        )
        # adam update
        updates, opt_state = optimizer_energy.update(grads, opt_state, params=x)
        x = optax.apply_updates(x, updates)

        new_carry = (opt_state, x, cell, angle)
        per_step_out = (loss_value, aux)

        return new_carry, per_step_out

    def step(carry, _):
        opt_state, x, cell, angle = carry
        # compute loss + grads
        (loss_value, aux), grads = jax.value_and_grad(scftRefine, has_aux=True)(
            x,
            cell,
            angle,
            f_param,
            sg,
            u0,
            id_class.dt,
            id_class.Ns,
            id_class.stencil,
            id_class.basis,
        )
        # adam update
        updates, opt_state = optimizer_refine.update(grads, opt_state, params=x)
        x = optax.apply_updates(x, updates)

        _, _, _, cell_stress, _, _ = aux

        cell = cell - cell_lr * cell_stress

        new_carry = (opt_state, x, cell, angle)
        per_step_out = (loss_value, aux)

        return new_carry, per_step_out

    # First force energy down (get out of disorder), then refine
    (opt_state_refine, x_opt, cell, angle), _ = jax.lax.scan(
        step_energy, (opt_state_refine, x_opt, cell, angle), None, length=20
    )
    (opt_state_refine, x_opt_out, cell_out, angle_out), (loss, aux) = jax.lax.scan(
        step,
        (opt_state_refine, x_opt, cell, angle),
        None,
        length=refine_epoch,
    )

    if angle.size > 0:
        x = (
            {
                "cell": cell_out,
                "angle": angle_out,
            }
            | x_opt_out
        )  # This is the refined target structure with cell and angle back in
    else:
        x = {"cell": cell_out} | x_opt_out

    return x, loss, aux


@jax.jit
def scft_eval(id_class, x_refine_no_grad, f_basis_no_grad, space_group, u0):
    reg_param = {"alpha": 1, "beta": 0, "gamma": 0, "target_energy": 0}
    f_param_no_grad = id_class.transform_fun(
        f_basis_no_grad
    )  # id_class.sequence.transform_fun(f_basis_no_grad)

    xi = x_refine_no_grad["xi"]
    psi = x_refine_no_grad["psi"]
    cell = x_refine_no_grad["cell"]
    angle = x_refine_no_grad["angle"] if "angle" in x_refine_no_grad else jnp.array([])
    loss, _ = scft(
        {"xi": xi, "psi": psi},
        cell,
        angle,
        f_param_no_grad,
        space_group,
        u0,
        reg_param,
        id_class.dt,
        id_class.Ns,
        id_class.stencil,
        id_class.basis,
    )  # Updates x_scftonly with the converged fields

    return loss


# @partial(jax.jit, static_argnames=['sg_target'])
def energy_loss(
    id_class,
    x_eval,
    x_alt_dict,
    alt_probability,
    alpha,
    beta,
    phi_target,
    sg_target,
    reg_param,
    prob_mixing,
):
    # Find the energy of the target
    # Outputs are: phi_hat, psi_hat, w_pred, incompress, cell_stress, F, F_disorder, L_ss
    # x_alt_dict only have the scft fields
    # Conver to block form
    f_basis = x_eval["f_basis"]
    f_param = id_class.transform_fun(
        f_basis
    )  # id_class.sequence.transform_fun(f_basis)
    x_eval["f_basis"] = f_param  # Update
    # evaluateEnergy puts f_fun through all the id_class-consistent equations
    F_converged, F_disorder = evaluateEnergyInverse(
        x_eval,
        list(id_class.space_group[sg_target].data.values()),
        list(id_class.space_group[sg_target].u0.values()),
        id_class.dt,
        id_class.Ns,
        id_class.stencil,
        id_class.basis,
    )

    # Recalculate energy for everything, even if not converged well - required to prevent loss jumping
    alt_energy = jnp.zeros((jnp.size(alt_probability),))
    for i in range(jnp.size(alt_probability)):
        # i is the index of the sampled structure, idx is the index of the space group
        key = list(id_class.sg_alt_dict.keys())[
            i
        ]  # Get the key of the alternate structure
        sg = id_class.sg_alt_dict[key]
        # Define new x_alt to have the current f_basis but converged alternate structures
        x_alt_eval = {"f_basis": f_param} | x_alt_dict[key]
        # Use the regular energy function for inverse design
        F_alt, _ = evaluateEnergyInverse(
            x_alt_eval,
            list(id_class.space_group[sg].data.values()),
            list(id_class.space_group[sg].u0.values()),
            id_class.dt,
            id_class.Ns,
            id_class.stencil,
            id_class.basis,
        )
        alt_energy = alt_energy.at[i].set(
            F_alt
        )  # Set the free energy of the evaluated structure

    # Calculate the alternate energy loss on ALL pass iterates of the energy
    alternate_energy_loss = energy_loss_fn(
        F_converged, alt_energy, reg_param["alt_offset"]
    )  # jnp.mean(jax.nn.relu(jnp.absolute(F_converged) / jnp.absolute(alt_energy) - (1 / reg_param['alt_offset'])))

    # Recalculate probabilities
    # To prevent a structure from being unsampled because it did have it's energy updated, do a mixing of the old and new probabilities
    new_alt_probability = calculate_probabilities(
        alt_energy, F_converged
    )  # Problem is, the less something is sampled if energy drifts down, the more likely it is not to be sampled again
    alt_probability = (
        prob_mixing * new_alt_probability + (1 - prob_mixing) * alt_probability
    )  # Mix old and new probabilities

    # phi deviation loss
    # Calculate the chemical identity field phi
    f_fun = jnp.dot(
        id_class.basis.T, f_param
    )  # basis_fun.T is Ns x Ncoeff, g is Ncoeff x Np
    g_fun = sequence_to_identity(f_fun)  # This is (Ns, )

    phi_converged = integral(
        x_eval["psi"] * g_fun, id_class.stencil
    )  # This is (Ntaus, 1)

    phi_dev_loss = jnp.mean((phi_converged - phi_target) ** 2)

    # Disorder energy losses
    energy_disorder_loss = energy_loss_fn(
        F_converged, F_disorder, reg_param["disorder_offset"]
    )  # jax.nn.relu(jnp.absolute(F_converged) / jnp.absolute(F_disorder) - (1 / reg_param['disorder_offset']))

    loss = alpha * phi_dev_loss + beta * (alternate_energy_loss + energy_disorder_loss)

    return (
        loss,
        (
            alt_energy,
            alt_probability,
            phi_dev_loss,
            alternate_energy_loss,
            energy_disorder_loss,
            F_converged,
            F_disorder,
        ),
    )


@partial(jax.jit, static_argnames=["sg_target"])
def calc_direction(
    id_class,
    x_loss,
    x_alt_dict,
    x_alt_list,
    x_alt_init_list,
    x_refine_no_grad_init,
    x_refine_no_grad,
    alt_probability,
    alpha,
    beta,
    epsilon,
    phi_target,
    sg_target,
    reg_param,
    prob_mixing,
    target_space_group,
    target_u0,
    alternate_space_group_list,
    alternate_u0_list,
):
    # x_loss and x_alt_dict must be the updated structure (id_class-consistent)

    # Pull out f_basis from x_loss - gradients are tracked
    f_basis = x_loss["f_basis"]
    output, grad_energy = jax.value_and_grad(energy_loss, has_aux=True, argnums=1)(
        id_class,
        x_loss,
        x_alt_dict,
        alt_probability,
        alpha,
        beta,
        phi_target,
        sg_target,
        reg_param,
        prob_mixing,
    )  # Single value
    loss_value, aux = output
    grad_energy_f = grad_energy[
        "f_basis"
    ]  # This is the gradient of the surrogate function with respect to f_basis

    grad_scft = jax.grad(scft_eval, argnums=2)(
        id_class, x_refine_no_grad_init, f_basis, target_space_group, target_u0
    )  # Tuple
    grad_scft_refine = jax.grad(scft_eval, argnums=2)(
        id_class, x_refine_no_grad, f_basis, target_space_group, target_u0
    )  # Single value
    # 1 is arg 2 and 0 is arg 1
    grad_q_hat_f = (
        grad_scft - grad_scft_refine
    )  # This is the gradient of the surrogate function with respect to f_basis

    # 3) Python‐level loop over alternates (static length ⇒ unrolled under jit)
    # Not necessary. we are not interested in preserving stability of the alternate structures
    for x_init, x_ref, sg, u0 in zip(
        x_alt_init_list, x_alt_list, alternate_space_group_list, alternate_u0_list
    ):
        g0 = jax.grad(scft_eval, argnums=2)(id_class, x_init, f_basis, sg, u0)
        g1 = jax.grad(scft_eval, argnums=2)(id_class, x_ref, f_basis, sg, u0)
        grad_q_hat_f = grad_q_hat_f + (g0 - g1)

    # Update the target fields
    # Calculate the single global kappa value
    dot = jnp.vdot(grad_energy_f.flatten(), grad_q_hat_f.flatten())
    norm2 = jnp.sqrt(jnp.vdot(grad_q_hat_f.flatten(), grad_q_hat_f.flatten()))
    kappa = jnp.fmax((epsilon - dot / (norm2 + 1e-8)), 0.0)

    scaled_dir_f = jnp.ones_like(grad_energy_f)
    dir_f = jax.tree_util.tree_map(
        lambda g_e, g_q: scaled_dir_f * (g_e + kappa * g_q),
        grad_energy_f,
        grad_q_hat_f,
    )
    dir_fields = jax.tree_util.tree_map(
        lambda v: jnp.zeros_like(v), x_refine_no_grad_init
    )

    grads = {"f_basis": dir_f, **dir_fields}

    return grads, loss_value, aux
