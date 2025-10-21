### Inverse design model
### For micelles

# Author: Oliver Xie, 2025

import jax
import jax.numpy as jnp
import optax
from pathlib import Path

from .scft_utils_micelle import scftID, scft, evaluateEnergyInverse, integral, simpsons_weights, scft, scftRefine, sequence_to_identity
from .save import SaveResults, TrajectoryRecorder, NSTrajectoryRecorder
from .scft_micelle import SCFTMicelle
from .id_utils import scft_refining_inner_loop, calc_direction, IDClass, calculate_probabilities, energy_loss_fn


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


class InverseDesignMicelle:
    def __init__(self, space_group, sequence, c):
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

        self.basis = sequence.basis_fun(self.s, Nbasis) # This stores the matrix basis rather than the function. We never need the function itself

        self.c = c # This is the overall polymer fraction (must be pre-defined)

    def load_target(self, sg: str, target_file : str, target_folder : Path):
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
            f_shape_slope = (1, self.sequence.Np) # Second row
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
        elif self.sequence.transform == "multiblock_poly":
            # Ndim is the number of dimensions, while Np is the number of blocks
            f_shape_frac = (1, self.sequence.Np)  # First row
            f_shape_coeff = (4, self.sequence.Np) # Hardcode a quartic polynomial. The coefficients are the same for all dimensions
            f_shape_mag = (self.sequence.Ndim, self.sequence.Np) # This is the zeroth order coefficient
            # Initialize all the block magnitudes to be between -2 and 2
            f_init_mag = jax.random.uniform(
                rng_key, minval=-2, maxval=2, shape=f_shape_mag
            )
            # Initialize all the coefficients between -2 and 2
            f_init_coeff = jax.random.uniform(
                rng_key, minval=-2, maxval=2, shape=f_shape_coeff
            )
            # We want all the block fractions to be more less equal. Remember this passes through a softmax
            f_init_frac = jax.random.uniform(
                rng_key, minval=-0.1, maxval=0.1, shape=f_shape_frac
            )
            f_init = jnp.vstack(
                (f_init_frac, f_init_coeff, f_init_mag) # Order is frac --> coeff (non-zeroth): 1 set for all dim --> mag (zeroth): Ndim
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

        # Generate an initial guess for the solvent identity
        solvent = jax.random.uniform(rng_key, minval = -2.0, maxval = 2.0, shape = (1,self.sequence.Ndim)) # Must be same dimension as polymer identity

        # Generate guesses for the field
        FT = self.space_group[sg].data["FT"]
        FS = self.space_group[sg].data["FS"]
        Ntaus = FT.shape[1]

        xi_shape = (Ntaus, 1)
        psi_shape = (Ntaus, self.Ns)
        solvent_shape = (Ntaus, 1)

        # Generate pseudo-self-consistent guesses (mean shifted)
        xi_init = jax.random.uniform(rng_key, shape=xi_shape)
        psi_init = jax.random.uniform(
            rng_key, shape=psi_shape, minval=1e-3, maxval=2
        )  # Always positive
        solvent_psi_init = jax.random.uniform(rng_key, shape=solvent_shape, minval=1e-3, maxval=2)

        # Zero average xi, and 1 average psi
        xi_star = jnp.matmul(FT, xi_init)
        xi_star = xi_star.at[0].set(0.0)
        xi_init = jnp.matmul(FS, xi_star)

        psi_star = jnp.matmul(FT, psi_init)
        psi_star = psi_star.at[0, :].set(self.c)
        psi_init = jnp.matmul(FS, psi_star)

        solvent_psi_star = jnp.matmul(FT, solvent_psi_init)
        solvent_psi_star = solvent_psi_star.at[0].set(1 - self.c)
        solvent_psi_init = jnp.matmul(FS, solvent_psi_star)

        # Generate guesses for cell and angle parameters
        # Guesses are the values contained in the cell_dict and angle_dict
        cell_guess = self.space_group[sg].cell
        angle_guess = self.space_group[sg].angle

        # Different checks for what x_init to return. Depends on if we have sequence or not, if we have angle or not
        if angle_guess.size == 0:
            # Has sequence, No angle
            x_init = {
                "f_basis": f_init,
                "solvent": solvent,
                "xi": xi_init,
                "psi": psi_init,
                "cell": cell_guess,
                "solvent_psi": solvent_psi_init,
            }
        elif angle_guess.size > 0:
            # Has sequence, Angle
            x_init = {
                "f_basis": f_init,
                "solvent": solvent,
                "xi": xi_init,
                "psi": psi_init,
                "cell": cell_guess,
                "angle": angle_guess,
                "solvent_psi": solvent_psi_init
            }
        
        return x_init

    def inverseDesignTarget(
        self, target, x, lr_param, reg_param, epoch_param
    ):
        # Extract the targets:
        target_phi = target["phi"] # The target density field
        target_sg = target["sg"] # The target space group

        # Set up learning rate parameters
        lr = lr_param["optimizer_lr"]
        seq_lr = lr_param["seq_lr"]
        cell_lr = lr_param["cell_lr"]

        # Set up optimizer
        param_labels = {"f_basis": "f_basis", "solvent": "solvent", "xi": "xi", "psi": "psi", "cell": "cell", "solvent_psi": "solvent_psi"}
        optimizer = optax.multi_transform(
            {
                "f_basis": optax.amsgrad(seq_lr),
                "solvent": optax.adam(seq_lr, nesterov=True),
                "xi": optax.adam(lr, nesterov=True),
                "psi": optax.adam(lr, nesterov=True),
                "cell": optax.adam(0.0, nesterov=True), # Turn off cell updates for now
                "solvent_psi": optax.adam(lr * 0.1, nesterov=True),
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
                "solvent": x["solvent"],
                "xi": x["xi"],
                "psi": x["psi"],
                "cell": x["cell"],
                "solvent_psi": x["solvent_psi"],
            }
            loss, aux = scftID(
                x_new,
                self.c,
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

    def run_all(self, target : dict, save_name : str):
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
        reg_param = {'phi': 1, 'scft': 1, 'energy': 1e-3, 'chebyshev': 0} # The mix of parameters to use here is an art
        epoch_param = {'epochs': 5000, 'print': 500, 'write': 1} # 5000
        lr_param = {'seq_lr': 0.1, 'optimizer_lr': 0.01, 'cell_lr': 0.01}
        x_id, id_recorder = self.inverseDesignTarget(
            target,
            x_init,
            lr_param,
            reg_param,
            epoch_param,
        )

        # 3. Refine target structure
        scft_run = SCFTMicelle(self.space_group, self.sequence, self.c, solvent = x_id['solvent'])
        print('Refining target structure')
        f_basis = self.sequence.transform_fun(x_id['f_basis']) # In multiblock function form
        solvent = x_id['solvent']
        x_id_noseq = x_id.copy()
        x_id_noseq.pop('f_basis', None)
        x_id_noseq.pop('solvent', None)

        total_epochs = 10000 # 10000
        epoch_param = {'epochs': total_epochs, 'print': 2000, 'grace' : 200}
        refine_param = {'lr_label': 'normal', 'field_init_value': 1e-2, 'cell_init_value': 5e-1, 'field_end_value': 1e-3, 'cell_end_value': 5e-2, 'transition': total_epochs, 'mix_transition': total_epochs}
        x_refine = scft_run.refinePhase(x_id_noseq, solvent, f_basis, target["sg"], lr_param = refine_param, epoch_param = epoch_param)

        # 4. Save
        data = {'id_x': x_id,
                'id_trajectory': id_recorder.trajectory,
                'x_refine': x_refine,
                'x_init': x_init}
        
        SaveResults().save_id(data, save_name)

        self.x_refine = x_refine
        self.x_id = x_id
        self.x_init = x_init

