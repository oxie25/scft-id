### Inverse design model

# Author: Oliver Xie, 2025

import jax
import jax.numpy as jnp
import optax
from pathlib import Path

from .scft_utils import (
    scftID,
    simpsons_weights,
)
from .save import SaveResults, TrajectoryRecorder, NSTrajectoryRecorder
from .scft import SCFT
from .id_utils import (
    scft_refining_inner_loop,
    calc_direction,
    IDClass,
    calculate_probabilities,
)


class EarlyStopping:
    def __init__(self, threshold, window=1000):
        self.percent_tolerance = threshold
        self.window = window
        self.history = jnp.zeros(window)
        self.index = 0
        self.full = False

    def check_stop(self, current_value):
        # Update the circular buffer
        self.history = self.history.at[self.index].set(current_value)
        self.index = (self.index + 1) % self.window
        if self.index == 0:
            self.full = True

        if self.full:
            # Calculate the maximum absolute percentage change in the window
            max_value = jnp.max(self.history)
            min_value = jnp.min(self.history)
            percent_change = (max_value - min_value) / max_value
            return percent_change < self.percent_tolerance
        else:
            return False


class InverseDesign:
    def __init__(self, space_group, sequence):
        # space group and sequence are classes
        # Store the space group and sequence
        self.space_group = space_group
        self.sequence = sequence

        # Generate sequence values
        # Get the Simpson's rule weights for the Ns dimension
        # Reshape the weights to make it a column vector so we can multiply
        # by A along the Ns dimension using matrix multiplication
        dt = sequence.dt
        Ns = sequence.Ns
        Nbasis = sequence.Nbasis
        stencil = simpsons_weights(Ns).reshape(Ns, 1) * dt  # Shape (Ns, 1)

        self.dt = dt
        self.stencil = stencil
        self.Ns = Ns
        self.s = jnp.linspace(0, 1, Ns)  # Ns points between 0 and 1

        self.basis = sequence.basis_fun(
            self.s, Nbasis
        )  # This stores the matrix basis rather than the function. We never need the function itself

    def load_target(self, sg: str, target_file: str, target_folder: Path):
        # Construct the target dictionary object
        phi_target = jnp.load(target_folder / target_file, allow_pickle=True)
        target = {"sg": sg, "phi": phi_target}
        return target

    def initial_guess(self, sg, rng_key=jax.random.key(0)):
        # Perform an initial guess of the sequence and desired structure fields - only for desired structure
        # Currently only two types of guesses are tailored: multiblock and tapered multiblock
        if self.sequence.transform == "multiblock":
            # Ndim is the number of dimensions, while Np is the number of blocks
            f_shape_frac = (1, self.sequence.Np)  # First row
            f_shape_mag = (self.sequence.Ndim, self.sequence.Np)
            # Initialize all the block magnitudes to be between -2 and 2
            f_init_mag = jax.random.uniform(
                rng_key, minval=-2, maxval=2, shape=f_shape_mag
            )
            # We want all the block fractions to be more less equal. Remember this passes through a softmax
            f_init_frac = jax.random.uniform(
                rng_key, minval=-0.1, maxval=0.1, shape=f_shape_frac
            )
            f_init = jnp.vstack(
                (f_init_frac, f_init_mag)
            )  # Frac is now first row, mag is second
        elif self.sequence.transform == "multiblock_taper":
            # Ndim is the number of dimensions, while Np is the number of blocks
            f_shape_frac = (1, self.sequence.Np)  # First row
            f_shape_slope = (1, self.sequence.Np)  # Second row
            f_shape_mag = (self.sequence.Ndim, self.sequence.Np)
            # Initialize all the block magnitudes to be between -2 and 2
            f_init_mag = jax.random.uniform(
                rng_key, minval=-2, maxval=2, shape=f_shape_mag
            )
            # Initialize all the slopes between -2 and 2
            f_init_slope = jax.random.uniform(
                rng_key, minval=-0.5, maxval=0.5, shape=f_shape_slope
            )
            # We want all the block fractions to be more less equal. Remember this passes through a softmax
            f_init_frac = jax.random.uniform(
                rng_key, minval=-0.1, maxval=0.1, shape=f_shape_frac
            )
            f_init = jnp.vstack(
                (f_init_frac, f_init_slope, f_init_mag)
            )  # Frac is now first row, mag is second
        else:
            # Here, f_shape is flipped from the above two cases
            # The guess is random numbers between -1 and 1 for each coefficient
            f_shape = (self.sequence.Np, self.sequence.Ndim)
            f_init = jax.random.uniform(rng_key, minval=-1, maxval=1, shape=f_shape)
            # Create a linear decay vector from 1 (no decay) to 0 (maximum decay)
            decay = jnp.linspace(1.0, 0.0, self.sequence.Np)
            # Apply decay to each row
            f_init = f_init * decay[:, None]
            # Set the first row explicitly to zero
            f_init = f_init.at[0].set(0.0)

        # Generate guesses for the field
        FT = self.space_group[sg].data["FT"]
        FS = self.space_group[sg].data["FS"]
        Ntaus = FT.shape[1]

        xi_shape = (Ntaus, 1)
        psi_shape = (Ntaus, self.Ns)

        # Generate pseudo-self-consistent guesses (mean shifted)
        xi_init = jax.random.uniform(rng_key, shape=xi_shape)
        psi_init = jax.random.uniform(
            rng_key, shape=psi_shape, minval=1e-3, maxval=2
        )  # Always positive

        # Zero average xi, and 1 average psi
        xi_star = jnp.matmul(FT, xi_init)
        xi_star = xi_star.at[0].set(0.0)
        xi_init = jnp.matmul(FS, xi_star)

        psi_star = jnp.matmul(FT, psi_init)
        psi_star = psi_star.at[0, :].set(1.0)
        psi_init = jnp.matmul(FS, psi_star)

        # Generate guesses for cell and angle parameters
        # Guesses are the values contained in the cell_dict and angle_dict
        cell_guess = self.space_group[sg].cell
        angle_guess = self.space_group[sg].angle

        # Different checks for what x_init to return. Depends on if we have sequence or not, if we have angle or not
        if angle_guess.size == 0:
            # Has sequence, No angle
            x_init = {
                "f_basis": f_init,
                "xi": xi_init,
                "psi": psi_init,
                "cell": cell_guess,
            }
        elif angle_guess.size > 0:
            # Has sequence, Angle
            x_init = {
                "f_basis": f_init,
                "xi": xi_init,
                "psi": psi_init,
                "cell": cell_guess,
                "angle": angle_guess,
            }

        return x_init

    def inverseDesignTarget(self, target, x, lr_param, reg_param, epoch_param):
        # Extract the targets:
        target_phi = target["phi"]  # The target density field
        target_sg = target["sg"]  # The target space group

        # Set up learning rate parameters
        lr = lr_param["optimizer_lr"]
        seq_lr = lr_param["seq_lr"]
        cell_lr = lr_param["cell_lr"]

        # Set up optimizer
        param_labels = {"f_basis": "f_basis", "xi": "xi", "psi": "psi", "cell": "cell"}
        optimizer = optax.multi_transform(
            {
                "f_basis": optax.amsgrad(seq_lr),
                "xi": optax.adam(lr, nesterov=True),
                "psi": optax.adam(lr, nesterov=True),
                "cell": optax.adam(cell_lr, nesterov=True),
            },
            param_labels,
        )

        opt_state = optimizer.init(x)

        # Set up space group values for target
        space_group = list(self.space_group[target_sg].data.values())
        u0 = list(self.space_group[target_sg].u0.values())

        # We need to transform from the sequence function parameter to the sequence function (if using block rep)
        @jax.jit
        def scftIDTransform(
            x, target_phi, space_group, u0_list, reg_param, dt, Ns, stencil, basis_fun
        ):
            # Generate the block function
            f_basis = x["f_basis"]
            f_fun = self.sequence.transform_fun(f_basis)

            # Remake the x that we send into the scft function
            x_new = {
                "f_basis": f_fun,
                "xi": x["xi"],
                "psi": x["psi"],
                "cell": x["cell"],
            }
            loss, aux = scftID(
                x_new,
                target_phi,
                space_group,
                u0_list,
                reg_param,
                dt,
                Ns,
                stencil,
                basis_fun,
            )

            return loss, aux

        # Jit the step of the optimizer
        @jax.jit
        def step(opt_state, x, reg_param=reg_param):
            output, grads = jax.value_and_grad(scftIDTransform, has_aux=True)(
                x,
                target_phi,
                space_group,
                u0,
                reg_param,
                self.dt,
                self.Ns,
                self.stencil,
                self.basis,
            )
            loss_value, aux = output

            # Update the learned variables
            updates, opt_state = optimizer.update(grads, opt_state, params=x)
            x = optax.apply_updates(x, updates)

            return opt_state, x, loss_value, aux

        # Store the loss curves per step to plot and see if we are going in right direction
        f_basis = x["f_basis"]
        num_epochs = epoch_param["epochs"]
        write_epochs = epoch_param["write"]

        write_size = num_epochs // write_epochs
        if num_epochs % write_epochs == 0:
            write_epochs += 0  # If divisible, last entry is the same as the last write
        else:
            write_epochs += (
                1  # If not divisible, last entry is the final converged result
            )

        recorder = TrajectoryRecorder(write_size, self.Ns, self.sequence.Ndim)

        # Learning
        for epoch in range(num_epochs):
            opt_state, x, loss_value, aux = step(opt_state, x, reg_param)
            (
                w_matlab,
                w_report,
                incompress_report,
                cell_stress,
                free_energy,
                free_energy_disorder,
                phi_hat,
                phi_report,
            ) = aux

            # Get the cell parameter(s)
            cell_temp = x["cell"]
            angle_temp = x["angle"] if "angle" in x else jnp.array([])

            # Calculate the sequence function to store
            f_basis = x["f_basis"]
            # f_basis = f_basis.at[1:].set(jax.nn.sigmoid(frac_sigmoid_factor * f_basis[1:]))
            # f_fun = jax.vmap(block, in_axes = 1)(f_basis).T
            f_fun = self.sequence.transform_fun(f_basis)

            # Store in array only every write
            if (epoch + 1) % write_epochs == 0:
                # Use recorder to record the current state
                recorder.record(
                    epoch=epoch,
                    loss_value=loss_value,
                    free_energy=free_energy,
                    free_energy_disorder=free_energy_disorder,
                    phi_hat=phi_hat,
                    cell=cell_temp,
                    w_report=w_report,
                    w_matlab=w_matlab,
                    incompress_report=incompress_report,
                    phi_report=phi_report,
                    f_fun=f_fun,
                )

            # Print loss
            if (epoch + 1) % epoch_param["print"] == 0:
                print(
                    f"Epoch {epoch + 1}, Loss: {loss_value:.8e}, Current Energy: {free_energy:.3f}, Disorder Energy: {free_energy_disorder:.3f}, Phi Target MAE: {phi_report:.5e}, Psi MAE: {w_report:.5e}, Incompress MAE: {incompress_report:.5e}, Cell Stress: {cell_stress}, Cell Param: {cell_temp}, Angle Param: {angle_temp}"
                )

        # Write one more time at exit if not divisible by write_epochs
        if num_epochs % write_epochs != 0:
            # Use recorder to record the current state
            recorder.record(
                epoch=epoch,
                loss_value=loss_value,
                free_energy=free_energy,
                free_energy_disorder=free_energy_disorder,
                phi_hat=phi_hat,
                cell=cell_temp,
                w_report=w_report,
                w_matlab=w_matlab,
                incompress_report=incompress_report,
                phi_report=phi_report,
                f_fun=f_fun,
            )

        return x, recorder

    def negativeSelection(
        self,
        target,
        x,
        x_alt_dict,
        sg_alt_dict,
        num_samples_desired,
        lr_param,
        reg_param,
        epoch_param,
    ):
        # We supply an already calculated x_alt_dict from SCFT using a designed sequence
        # Use the bilevel optimization method introduced in BOME! paper for block copolymers
        # in this approach, the function and ALL fields must be optimized at the same time (including alternate structure fields)

        # Extract the targets:
        target_sg = target["sg"]

        # Set up an SCFT model
        scft_model = SCFT(self.space_group, self.sequence)

        # Evaluate the alternate phases for their free energies and calculate probability
        f_basis = x["f_basis"]
        f_fun = self.sequence.transform_fun(f_basis)
        x_eval = x.copy()  # Need to use the converted basis
        x_eval["f_basis"] = f_fun
        p_alt = scft_model.evaluatePhase(x=x_eval, sg=target_sg)
        F_design = p_alt["energy"]  # This is the free energy of the designed phase

        alt_energy = jnp.zeros((len(sg_alt_dict),))  # a vector of all the structures
        i = 0
        for phase in sg_alt_dict.keys():
            x_eval = {"f_basis": f_fun} | x_alt_dict[phase]
            p_alt = scft_model.evaluatePhase(x=x_eval, sg=sg_alt_dict[phase])
            alt_energy = alt_energy.at[i].set(p_alt["energy"])
            i += 1

        alt_probability = calculate_probabilities(alt_energy, F_design)

        # Set up some used parameters
        alpha = reg_param["phi"]
        beta = reg_param["energy"]
        prob_mixing = reg_param["probability"]
        epsilon = reg_param["epsilon"]  # For the BOME method
        # We should test different optimizers
        # Set up optimizer and opt_state
        seq_lr = lr_param["sequence_lr"]
        refine_epoch = epoch_param["refine"]["epochs"]
        num_samples = min(
            num_samples_desired, len(sg_alt_dict)
        )  # Must be less than the number of alternate structures
        num_epochs = epoch_param["epochs"]

        # Only change sequence with a warm-up schedule for f_basis
        param_labels = {"f_basis": "f_basis", "xi": "xi", "psi": "psi", "cell": "cell"}
        warmup_epochs = lr_param["warmup_epochs"]  # Number of warm-up epochs
        final_seq_lr = seq_lr  # Final learning rate for f_basis after warm-up
        f_basis_lr_schedule = optax.linear_schedule(
            init_value=1e-3 * final_seq_lr,
            end_value=final_seq_lr,
            transition_steps=warmup_epochs,
        )

        # Set up optimizer
        # Use AMSGrad for f_basis, zero out gradients for xi, psi, and cell (we let them be their SCF values)
        optimizer = optax.multi_transform(
            {
                "f_basis": optax.amsgrad(f_basis_lr_schedule),
                "xi": optax.set_to_zero(),
                "psi": optax.set_to_zero(),
                "cell": optax.set_to_zero(),
            },
            param_labels,
        )
        opt_state_target = optimizer.init(x)  # x_seq or x

        # Set up recorder for trajectory
        recorder = NSTrajectoryRecorder(num_epochs, self.Ns, self.sequence.Ndim)

        # Define a helper class that is passable into the jax.jit calc_direction
        id_class = IDClass(
            dt=self.dt,
            Ns=self.Ns,
            stencil=self.stencil,
            basis=self.basis,
            space_group=self.space_group,
            transform_fun=self.sequence.transform_fun,
            sg_alt_dict=sg_alt_dict,
        )

        # This is the main loop, performing steps in f only
        def step_Alternate(
            opt_state_target,
            x,
            x_alt_dict,
            alt_energy,
            alt_probability,
            alpha,
            beta,
            epsilon,
            refine_epoch,
            num_samples,
            rng_key,
            lr_param,
        ):
            # Create a copy of x with f_basis removed
            x_refine_init = x.copy()
            x_refine_init.pop("f_basis", None)

            # Keep f_basis_no_grad
            f_basis_no_grad = jax.lax.stop_gradient(x["f_basis"])

            # Refine the target and alternates - no gradient tracking
            space_group = list(self.space_group[target_sg].data.values())
            u0 = list(self.space_group[target_sg].u0.values())

            x_refine_no_grad, loss_value_refining, aux = jax.lax.stop_gradient(
                scft_refining_inner_loop(
                    id_class,
                    x_refine_init.copy(),
                    f_basis_no_grad,
                    space_group,
                    u0,
                    lr_param["refine"]["field_init_value"],
                    lr_param["refine"]["cell_init_value"],
                    refine_epoch,
                )
            )
            print(f"Finished refining {target_sg}. Loss is {loss_value_refining[-1]}")
            # Refine alternates based on random sampling
            x_alt_dict_init = x_alt_dict.copy()  # Copy the initial dictionary
            # Randomly sample num_samples from the alternate structures
            num_alt_structures = len(sg_alt_dict)
            sampled_sg_idx = jax.random.choice(
                rng_key,
                num_alt_structures,
                shape=(num_samples,),
                replace=False,
                p=alt_probability,
            )

            x_alt_list = []
            x_alt_init_list = []
            alternate_space_group_list = []
            alternate_u0_list = []
            alternate_loss = jnp.zeros(len(sg_alt_dict))

            for i in sampled_sg_idx:
                # i is the index of the sampled structure, idx is the index of the space group
                key = list(sg_alt_dict.keys())[i]
                sg = sg_alt_dict[key]
                space_group = list(self.space_group[sg].data.values())
                u0 = list(self.space_group[sg].u0.values())
                # Use the regular energy function for inverse design
                x_alt_refine, loss_value, aux = jax.lax.stop_gradient(
                    scft_refining_inner_loop(
                        id_class,
                        x_alt_dict_init[key].copy(),
                        f_basis_no_grad,
                        space_group,
                        u0,
                        lr_param["refine"]["field_init_value"],
                        lr_param["refine"]["cell_init_value"],
                        refine_epoch,
                    )
                )  # Refines the alternate structure
                x_alt_dict[key] = (
                    x_alt_refine  # Update the dictionary with the refined structure
                )
                print(f"Finished refining {sg}. Loss is {loss_value[-1]}")

                # Generate the necessary lists to pass to the body function
                x_alt_list.append(x_alt_dict[key])
                x_alt_init_list.append(x_alt_dict_init[key])
                alternate_space_group_list.append(
                    list(self.space_group[sg].data.values())
                )
                alternate_u0_list.append(list(self.space_group[sg].u0.values()))
                alternate_loss = alternate_loss.at[i].set(loss_value[-1])

            # This calculates the loss and gradients through free energy (and phi deviation) only
            x_loss = {"f_basis": f_basis} | x_refine_no_grad
            target_space_group = list(self.space_group[target_sg].data.values())
            target_u0 = list(self.space_group[target_sg].u0.values())

            grads, loss_value, aux = calc_direction(
                id_class,
                x_loss,
                x_alt_dict,
                x_alt_list,
                x_alt_init_list,
                x_refine_init,
                x_refine_no_grad,
                alt_probability,
                alpha,
                beta,
                epsilon,
                target["phi"],
                target["sg"],
                reg_param,
                prob_mixing,
                target_space_group,
                target_u0,
                alternate_space_group_list,
                alternate_u0_list,
            )

            # print(grads['f_basis'])

            alt_energy, alt_probability, _, energy_loss, alt_energy_loss, F, *_ = aux
            print(
                f"The calculated energy for the target is {F}, for alternate structures are {alt_energy}"
            )

            # Check if we want to make next step - current criteria is energy losses only
            if energy_loss <= 0.0 and alt_energy_loss <= 0.0:
                # Exit without updating x
                print(
                    f"Loss is {loss_value}, stop criteria reached, exiting without updating x"
                )
                return (
                    opt_state_target,
                    x,
                    x_alt_dict,
                    loss_value,
                    aux,
                    jnp.array(alternate_loss),
                    loss_value_refining[-1],
                )

            # Update the learned variables
            updates, opt_state_target = optimizer.update(
                grads, opt_state_target, params=x
            )  # or x
            x = {
                "f_basis": optax.apply_updates(x, updates)["f_basis"],
                **x_refine_no_grad,
            }

            return (
                opt_state_target,
                x,
                x_alt_dict,
                loss_value,
                aux,
                jnp.array(alternate_loss),
                loss_value_refining[-1],
            )

        # Learning
        for epoch in range(num_epochs):
            # Set up a different random key for each epoch. This is to choose alt structures
            stochastic_key = jax.random.PRNGKey(epoch)

            (
                opt_state_target,
                x,
                x_alt_dict,
                loss_value,
                aux,
                alternate_loss,
                loss_refining,
            ) = step_Alternate(
                opt_state_target,
                x,
                x_alt_dict,
                alt_energy,
                alt_probability,
                alpha,
                beta,
                epsilon,
                refine_epoch,
                num_samples,
                stochastic_key,
                lr_param,
            )
            (
                alt_energy,
                alt_probability,
                phi_dev_loss,
                alternate_energy_loss,
                energy_disorder_loss,
                F_target,
                F_disorder,
            ) = aux

            # Calculate the sequence function to store
            f_basis = x["f_basis"]
            f_param = self.sequence.transform_fun(f_basis)
            f_fun = jnp.dot(self.basis.T, f_param)

            # Store in recorder instead of separate arrays
            recorder.record(
                epoch=epoch,
                target_energy=F_target,
                disorder_energy=F_disorder,
                sequence=f_fun,
                target_refining_loss=loss_refining,
                alternate_energy=alt_energy,
                alternate_loss=alternate_loss,
                target_loss=loss_value,
            )

            # Print loss
            if (epoch + 1) % epoch_param["print"] == 0:
                print(
                    f"Main Convergence Loop: Epoch {epoch + 1}, Loss: {loss_value:.8f}, Phi Target Loss: {phi_dev_loss:.5f}, Energy Loss: {alternate_energy_loss:.5f}, Energy Disorder Loss: {energy_disorder_loss:.5f}, Free Energy {F_target:.3f}, Disorder Energy: {F_disorder:.3f}"
                )

            if alternate_energy_loss <= 0.0 and energy_disorder_loss <= 0.0:
                # The x returned by step_Alternate should be the x that gave the print-out solutions.
                print(
                    f"Energy losses below zero, stopped on {epoch + 1}: Loss: {loss_value:.8f}, Phi Target Loss: {phi_dev_loss:.5f}, Energy Loss: {alternate_energy_loss:.5f}, Energy Disorder Loss: {energy_disorder_loss:.5f}, Free Energy {F_target:.3f}, Disorder Energy: {F_disorder:.3f}"
                )
                break

        return x, x_alt_dict, recorder

    def run_all(
        self,
        target: dict,
        alt_sg: list,
        num_samples_desired: int,
        save_name: str,
        ns_iter: int = 50,
        ns_offset: float = 1.0005,
    ):
        # Run all the methods in order
        # Inputs: target supplies a target structure and space group as a dict
        #       alt_sg supplies a list of alternate space groups that should be investigated
        #       num_samples_desired is the number of alternate structures to sample
        #       save_name is the filename to save results

        # 1. Make an initial guess
        print(f" Making initial guess for target structure {target['sg']}")
        x_init = self.initial_guess(target["sg"])

        # 2. Inverse design the target structure
        print(f"Inverse designing target structure {target['sg']}")
        reg_param = {
            "phi": 1,
            "scft": 1,
            "energy": 0,
            "chebyshev": 0,
        }  # The mix of parameters to use here is an art
        epoch_param = {"epochs": 5000, "print": 500, "write": 1}  # 5000
        lr_param = {"seq_lr": 0.1, "optimizer_lr": 0.01, "cell_lr": 0.01}
        x_id, id_recorder = self.inverseDesignTarget(
            target,
            x_init,
            lr_param,
            reg_param,
            epoch_param,
        )

        # 3. Refine target structure
        scft_run = SCFT(self.space_group, self.sequence)
        print("Refining target structure")
        f_basis = self.sequence.transform_fun(
            x_id["f_basis"]
        )  # In multiblock function form
        x_id_noseq = x_id.copy()
        x_id_noseq.pop("f_basis", None)

        total_epochs = 10000  # 10000
        epoch_param = {"epochs": total_epochs, "print": 2000, "grace": 200}
        refine_param = {
            "lr_label": "normal",
            "field_init_value": 1e-2,
            "cell_init_value": 5e-1,
            "field_end_value": 1e-3,
            "cell_end_value": 5e-2,
            "transition": total_epochs,
            "mix_transition": total_epochs,
        }
        x_refine = scft_run.refinePhase(
            x_id_noseq,
            f_basis,
            target["sg"],
            lr_param=refine_param,
            epoch_param=epoch_param,
        )

        # 4. Converge alternate structures given the designed sequence
        # Make initial star dict by looping over alt_sg
        initial_star_dict = {sg: self.space_group[sg].initial_star for sg in alt_sg}

        x_alt_dict, sg_alt_dict = scft_run.initialGuess(
            f_basis, alt_sg, initial_star_dict, 0.02
        )  # 0.1 originally, 0.02 works well too

        total_epochs = 10000  # 10000
        lr_param = {
            "lr_label": "normal",
            "field_init_value": 1e-2,
            "cell_init_value": 5e-1,
            "field_end_value": 1e-3,
            "cell_end_value": 5e-2,
            "transition": total_epochs,
            "mix_transition": total_epochs,
        }

        # Refine all alternate structures
        i = 0
        epoch_param = {
            "first_optimization": 200,
            "second_optimization": 200,
            "third_optimization": total_epochs,
            "grace": 200,
            "print": 2000,
        }
        F_target = 0

        loss_param = {"scft_weight": 1, "free_energy_weight": 0, "energy_dif_weight": 1}

        for phase in sg_alt_dict.keys():
            print(f"Converging alternate structure: {phase}")
            x_temp = scft_run.convergePhase(
                x_alt_dict[phase],
                f_basis,
                sg_alt_dict[phase],
                F_target,
                lr_param,
                epoch_param=epoch_param,
                loss_param=loss_param,
            )
            print(f"Optimization complete, writing into index {i}")
            x_alt_dict[phase] = x_temp

        # 5. Negative selection against alternates
        reg_param = {
            "phi": 0,
            "energy": 1,
            "probability": 0.2,
            "epsilon": 0.1,
            "alt_offset": ns_offset,
            "disorder_offset": 1.002,
        }  # if we want to strictly adhere to the target structure, set as (1, 1). If we have no idea if target is feasible/lowest energy, and only care about space group, set as (0, 1). Probablity is the weight of the new probability to mix with the old one. offset > 1 is percentage we want to be below the compared energy. Error_stall is in percent
        lr_param = {
            "warmup_epochs": 2,
            "sequence_lr": 0.1,
            "field_lr": 0.0,
            "error_stall": 4,
            "patience_stall": 5,
            "refine": {
                "lr_label": "normal",
                "field_init_value": 1e-2,
                "cell_init_value": 5e-1,
                "field_end_value": 1e-2,
                "cell_end_value": 1e-1,
                "transition": 1000,
                "mix_transition": 1000,
            },
        }
        epoch_param = {
            "epochs": ns_iter,
            "print": 1,
            "refine": {"epochs": 1000, "print": 1000, "grace": 20},
        }  # Default 50

        # We need to rebuild x with the parameters
        x = {"f_basis": f_basis} | x_refine
        x_id_neg = {**x, "f_basis": x_id["f_basis"]}
        x_negative, x_alt_dict_negative, ns_recorder = self.negativeSelection(
            target,
            x_id_neg.copy(),
            x_alt_dict.copy(),
            sg_alt_dict,
            num_samples_desired,
            reg_param=reg_param,
            lr_param=lr_param,
            epoch_param=epoch_param,
        )

        # 6. Refine all structures again
        print("After negative selection, refining target structure")
        f_basis = x_negative["f_basis"]
        f_param = self.sequence.transform_fun(f_basis)

        if "angle" in x_negative:
            x_negative_noseq = {
                "xi": x_negative["xi"],
                "psi": x_negative["psi"],
                "cell": x_negative["cell"],
                "angle": x_negative["angle"],
            }
        else:
            x_negative_noseq = {
                "xi": x_negative["xi"],
                "psi": x_negative["psi"],
                "cell": x_negative["cell"],
            }

        total_epoch = 5000  # 5000
        refine_param = {
            "lr_label": "normal",
            "field_init_value": 5e-4,
            "cell_init_value": 1e-1,
            "field_end_value": 5e-4,
            "cell_end_value": 1e-1,
            "transition": total_epoch,
            "mix_transition": total_epoch,
        }
        epoch_param = {"epochs": total_epoch, "print": 2500, "grace": 200}
        x_negative_refine = scft_run.refinePhase(
            x_negative_noseq.copy(),
            f_param,
            target["sg"],
            lr_param=refine_param,
            epoch_param=epoch_param,
        )

        # Refine and evaluate all alternate structures
        x_alt_dict_negative_refine = x_alt_dict_negative.copy()  # Create new instance
        # Refine all alternate structures
        i = 0
        for phase in sg_alt_dict.keys():
            print(f"Refining alternate structure: {phase}")
            x_temp = scft_run.refinePhase(
                x_alt_dict_negative[phase].copy(),
                f_param,
                sg_alt_dict[phase],
                lr_param=refine_param,
                epoch_param=epoch_param,
            )
            x_alt_dict_negative_refine[phase] = x_temp
            print(f"Optimization complete, writing into index {i}")
            i += 1

        # 7. Save
        data = {
            "id_x": x_id,
            "id_trajectory": id_recorder.trajectory,
            "negative_x": x_negative,
            "negative_trajectory": ns_recorder.trajectory,
            "negative_alt_x": x_alt_dict_negative,
            "negative_refine_x": x_negative_refine,
            "negative_refine_alt_x": x_alt_dict_negative_refine,
            "x_refine": x_refine,
            "x_alt_dict": x_alt_dict,
        }

        SaveResults().save_id(data, save_name)
