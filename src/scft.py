import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
from .scft_utils import (
    simpsons_weights,
    scft,
    scftRefine,
    evaluateFields,
    sequence_to_identity,
    evaluateEnergyInverse,
)
from .structures_utils import field_initialize, field_star_initialize
from .save import SaveResults


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


# This class runs SCFT calculations for a set of desired space groups
class SCFT:
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

    def shiftXi(self, xi, sg):
        FT = self.space_group[sg].data["FT"]
        FS = self.space_group[sg].data["FS"]

        # Mean shift xi because xi_star[0] = 0
        xi_star = jnp.matmul(FT, xi)
        xi_star = xi_star.at[0].set(0.0)
        xi = jnp.matmul(FS, xi_star)

        return xi

    def evaluatePhase(self, x, sg):
        # Evaluate the current phase and return a set of parameters including deviations in SCFT equations, free energy, free energy of disordered phase, cell parameters
        space_group = self.space_group[sg].data
        u0_dict = self.space_group[sg].u0

        p = evaluateFields(
            x,
            list(space_group.values()),
            list(u0_dict.values()),
            self.dt,
            self.Ns,
            self.stencil,
            self.basis,
        )

        return p

    def evaluateEnergy(self, x, sg):
        # Evaluate the current phase and return a set of parameters including deviations in SCFT equations, free energy, free energy of disordered phase, cell parameters
        space_group = self.space_group[sg].data
        u0_dict = self.space_group[sg].u0

        E, E_dis = evaluateEnergyInverse(
            x,
            list(space_group.values()),
            list(u0_dict.values()),
            self.dt,
            self.Ns,
            self.stencil,
            self.basis,
        )

        return E, E_dis

    def initialGuess(self, f_basis, sg_list, initial_star_dict, strength):
        # Generate an initial guess given f_basis (the transformed sequence parameters)
        # There are always two initial guesses for each space group, one with g_normal and one with g_flip
        f_fun = jnp.dot(self.basis.T, f_basis)
        g_fun = sequence_to_identity(f_fun)

        # Costruct the necessary guesses and parameter tuples/arrays for all alternate space groups.
        # The output of the function are two lists, one is x_guess and the other is param
        # Iterate through the space_group list which contains all loaded space groups and the calculated parameters
        combinations = [
            ["g_normal", "positive"],
            ["g_normal", "negative"],
        ]  # These now cover all possibilities, ['g_flip', 'positive'], ['g_flip', 'negative']]

        x_dict = {}  # This stores the initial guesses of xi, psi, cell, and angle for each space group
        sg_dict = {}  # This stores the name of the space group for each initial guess

        for sg in sg_list:
            sg_param = self.space_group[sg].data  # Access the data
            FT = sg_param["FT"]
            FS = sg_param["FS"]

            Ntaus = FS.shape[0]
            Nstars = FT.shape[0]
            # Mean shift
            xi_field = field_initialize(Ntaus, 1)
            # Transform a few times to get the right zeroed values
            xi_star = jnp.matmul(FT, xi_field)
            xi_star = xi_star.at[0].set(0.0)
            xi_field = jnp.matmul(FS, xi_star)
            for j, combo in enumerate(combinations):
                key = f"{sg}_{j}"  # Key for the dictionary
                psi_field = field_star_initialize(
                    combo[0],
                    combo[1],
                    initial_star_dict[sg],
                    Nstars,
                    self.Ns,
                    FS,
                    FT,
                    g_fun,
                    value=strength,
                )
                # Construct initial guess tuple
                # If there is an angle, then must include for initial guess
                if self.space_group[sg].angle.size > 0:
                    x_dict[key] = {
                        "xi": xi_field,
                        "psi": psi_field,
                        "cell": self.space_group[sg].cell,
                        "angle": self.space_group[sg].angle,
                    }
                else:
                    x_dict[key] = {
                        "xi": xi_field,
                        "psi": psi_field,
                        "cell": self.space_group[sg].cell,
                    }

                sg_dict[key] = sg

        self.x_dict = x_dict  # Store as initial guess of alternate structures
        self.sg_dict = sg_dict  # These correspond to the row of space group, cell param, and angle param to use when investigating alternate space groups

        return x_dict, sg_dict

    def loadPhase(self, psi_filename, sg_key, cell_array, angle_array=None):
        # Load a phase. Identify if it is ending in .csv or .npy and load appropriately
        x_dict = {}  # This stores the initial guesses of xi, psi, cell, and angle for each space group
        sg_dict = {}  # This stores the name of the space group for each initial guess

        for sg in sg_key:
            sg_param = self.space_group[sg].data  # Access the data
            FT = sg_param["FT"]
            FS = sg_param["FS"]

            Ntaus = FS.shape[0]
            Nstars = FT.shape[0]
            # Mean shift
            xi_field = field_initialize(Ntaus, 1)
            # Transform a few times to get the right zeroed values
            xi_star = jnp.matmul(FT, xi_field)
            xi_star = xi_star.at[0].set(0.0)
            xi_field = jnp.matmul(FS, xi_star)
            # Load the guess
            ext = os.path.splitext(psi_filename)[1].lower()
            if ext == ".npy":
                data = np.load(psi_filename)
            elif ext == ".csv":
                data = np.loadtxt(psi_filename, delimiter=",")
            else:
                raise ValueError(f"Unsupported file extension: {ext}")

            # data is star, do a transform back to tau.
            psi_field = jnp.matmul(FS, jnp.array(data))

            # Construct initial guess tuple
            # If there is an angle, then must include for initial guess
            if angle_array is not None:
                x_dict[sg] = {
                    "xi": xi_field,
                    "psi": psi_field,
                    "cell": cell_array,
                    "angle": angle_array,
                }
            else:
                x_dict[sg] = {
                    "xi": xi_field,
                    "psi": psi_field,
                    "cell": cell_array,
                }

            sg_dict[sg] = sg

        self.x_dict = x_dict  # Store as initial guess of alternate structures
        self.sg_dict = sg_dict  # These correspond to the row of space group, cell param, and angle param to use when investigating alternate space groups

        return x_dict, sg_dict

    def convergePhase(
        self,
        x,
        f_basis,
        sg,
        F_target,
        lr_param,
        epoch_param,
        loss_param,
    ):
        # Converge a self-consistent solution given an initial guess x, the sequence function f_basis, the space group sg, and the target free energy F_target
        # lr_param, epoch_param, and loss_param are dictionaries containing the learning rate, number of epochs, and loss parameters respectively

        # x should contain xi, psi, cell, and angle parameters
        # Always do explicit gradient descent of cell and angle parameters

        # Three step convergence.
        # 1. Run scftNoStress with penalty of energy comparison term
        # 2. Run scftNoStress without penalty of energy comparison term
        # 3. Run scftNorm without penalty of energy comparison term

        # We want to remove cell and stress from x so no differentiation graph is calculated for them. Especially important if we are using V * incompress as a loss
        x_opt = {"xi": x["xi"], "psi": x["psi"]}
        cell = x["cell"]
        angle = x["angle"] if "angle" in x else jnp.array([])

        ### Step 1: Minimizing free energy while constraining via SCFT equations
        print("First optimization: minimizing free energy")
        optimizer = optax.adam(learning_rate=0.05, nesterov=True)
        opt_state = optimizer.init(x_opt)

        # Unpack required space group, u0_list
        space_group = list(self.space_group[sg].data.values())
        u0 = list(self.space_group[sg].u0.values())

        alpha = loss_param["scft_weight"]
        beta = loss_param["free_energy_weight"]
        gamma = loss_param["energy_dif_weight"]

        reg_param = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "target_energy": F_target,
        }

        @jax.jit
        def step_LowEnergy(
            opt_state,
            x,
            cell,
            angle,
            f_basis,
            space_group=space_group,
            u0_list=u0,
            reg_param=reg_param,
            dt=self.dt,
            Ns=self.Ns,
            stencil=self.stencil,
            basis_fun=self.basis,
        ):
            output, grads = jax.value_and_grad(scft, has_aux=True)(
                x,
                cell,
                angle,
                f_basis,
                space_group,
                u0_list,
                reg_param,
                dt,
                Ns,
                stencil,
                basis_fun,
            )
            loss_value, aux = output

            # Update the SCFT fields - not unit cell parameters
            updates, opt_state = optimizer.update(grads, opt_state, params=x)
            x = optax.apply_updates(x, updates)

            return opt_state, x, loss_value, aux

        for epoch in range(epoch_param["first_optimization"]):
            # Compute the loss and gradients
            opt_state, x_opt, loss_value, aux = step_LowEnergy(
                opt_state, x_opt, cell, angle, f_basis
            )
            (
                w_matlab,
                w_report,
                incompress_report,
                cell_stress,
                free_energy,
                free_energy_disorder,
            ) = aux

            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}, Loss: {loss_value:.6e}, Free Energy: {free_energy:.3f}, Incompressible Loss: {incompress_report:.6e}, W Loss: {w_report:.6e}, W MATLAB Criteria: {w_matlab:.6e}, Stress Loss: {cell_stress}, Cell: {cell}"
                )

        ### Step 2
        print("Second optimization: refining structure without stress")
        optimizer = optax.adam(learning_rate=0.01, nesterov=True)
        opt_state = optimizer.init(x_opt)

        # Update regularization parameters
        reg_param["alpha"] = 1
        reg_param["beta"] = 0
        reg_param["gamma"] = 0

        @jax.jit
        def step_NoStress(
            opt_state,
            x,
            cell,
            angle,
            f_basis,
            space_group=space_group,
            u0_list=u0,
            reg_param=reg_param,
            dt=self.dt,
            Ns=self.Ns,
            stencil=self.stencil,
            basis_fun=self.basis,
        ):
            output, grads = jax.value_and_grad(scft, has_aux=True)(
                x,
                cell,
                angle,
                f_basis,
                space_group,
                u0_list,
                reg_param,
                dt,
                Ns,
                stencil,
                basis_fun,
            )
            loss_value, aux = output

            # Update the SCFT fields - not unit cell parameters
            updates, opt_state = optimizer.update(grads, opt_state, params=x)
            x = optax.apply_updates(x, updates)

            return opt_state, x, loss_value, aux

        for epoch in range(epoch_param["second_optimization"]):
            # Compute the loss and gradients
            opt_state, x_opt, loss_value, aux = step_NoStress(
                opt_state, x_opt, cell, angle, f_basis
            )
            (
                w_matlab,
                w_report,
                incompress_report,
                cell_stress,
                free_energy,
                free_energy_disorder,
            ) = aux

            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}, Loss: {loss_value:.6e}, Free Energy: {free_energy:.3f}, Incompressible Loss: {incompress_report:.6e}, W Loss: {w_report:.6e}, W MATLAB Criteria: {w_matlab:.6e}, Stress Loss: {cell_stress}, Cell: {cell}"
                )

        ### Step 3
        # Relax unit cell using gradient descent
        # Use direct gradient descent on cell and angle param
        print(
            "Third optimization: refining structure with gradient descent on cell param using stress"
        )
        # All learnings rates are on a schedule. Setting the beginning and end values equivalent makes it a constant learning rate
        # lr for the non-cell parameters
        lr = optax.linear_schedule(
            init_value=lr_param["field_init_value"],
            end_value=lr_param["field_end_value"],
            transition_steps=lr_param["transition"],
        )
        # lr for the cell parameters
        cell_lr = jnp.linspace(
            lr_param["cell_init_value"],
            lr_param["cell_end_value"],
            lr_param["transition"],
        )
        # Append constant end value if transition is less than the total number of epochs
        remaining_steps = epoch_param["third_optimization"] - lr_param["transition"]
        cell_lr = jnp.concatenate(
            [cell_lr, lr_param["cell_end_value"] * jnp.ones(remaining_steps)]
        )

        optimizer = optax.adam(learning_rate=lr, nesterov=True)
        opt_state = optimizer.init(x_opt)

        @jax.jit
        def step_Stress(
            opt_state,
            x,
            cell,
            angle,
            f_basis,
            space_group=space_group,
            u0_list=u0,
            dt=self.dt,
            Ns=self.Ns,
            stencil=self.stencil,
            basis_fun=self.basis,
        ):
            output, grads = jax.value_and_grad(scftRefine, has_aux=True)(
                x,
                cell,
                angle,
                f_basis,
                space_group,
                u0_list,
                dt,
                Ns,
                stencil,
                basis_fun,
            )
            loss_value, aux = output

            # Update the SCFT fields - not unit cell parameters
            updates, opt_state = optimizer.update(grads, opt_state, params=x)
            x = optax.apply_updates(x, updates)

            return opt_state, x, loss_value, aux

        early_stopping = EarlyStopping(threshold=1e-5)
        grace_period = epoch_param["grace"]  # Don't check for x epochs

        for epoch in range(epoch_param["third_optimization"]):
            # Compute the loss and gradients
            opt_state, x_opt, loss_value, aux = step_Stress(
                opt_state, x_opt, cell, angle, f_basis
            )
            (
                w_matlab,
                w_report,
                incompress_report,
                cell_stress,
                free_energy,
                free_energy_disorder,
            ) = aux

            if epoch + 1 > grace_period:
                # Manually update cell with stress
                cell = cell - cell_lr[epoch] * cell_stress

            if (epoch + 1) % epoch_param["print"] == 0:
                print(
                    f"Epoch {epoch + 1}, Loss: {loss_value:.6e}, Free Energy: {free_energy:.3f}, Incompressible Loss: {incompress_report:.6e}, W Loss: {w_report:.6e}, W MATLAB Criteria: {w_matlab:.6e}, Stress Loss: {cell_stress}, Cell: {cell}"
                )

            # Be careful with stoppping criteria. For 3D structures a lower one is sufficient, for 1D structures a higher one is required
            if loss_value <= 1e-12:
                print(
                    f"Early stopping due to below tolerance, last epoch: Epoch {epoch + 1}, Loss: {loss_value:.6e}, Free Energy: {free_energy:.3f}, Incompressible Loss: {incompress_report:.6e}, W Loss: {w_report:.6e}, Stress Loss: {cell_stress}, Cell: {cell}"
                )
                break

            # Early stopping is on the energy criteria
            if (early_stopping.check_stop(jnp.array(free_energy))) and (
                epoch >= grace_period
            ):
                print(
                    f"Early stopping due to energy difference in window below threshold, last epoch: Epoch {epoch + 1}, Loss: {loss_value:.6e}, Free Energy: {free_energy:.3f}, Incompressible Loss: {incompress_report:.6e}, W Loss: {w_report:.6e}, Stress Loss: {cell_stress}, Cell: {cell}"
                )
                break

        # Update the stored converged phases - We can only freely shift the xi field. Do so because in scftRefine this occurs
        # Reconstruct the fields with regressed values
        if angle.size > 0:
            # shift the xi field
            # Mean shift xi because xi_star[0] = 0
            xi = self.shiftXi(x_opt["xi"], sg)
            x = {"xi": xi, "psi": x_opt["psi"], "cell": cell, "angle": angle}
        else:
            xi = self.shiftXi(x_opt["xi"], sg)
            x = {"xi": xi, "psi": x_opt["psi"], "cell": cell}
        # Return the converged phase
        return x

    def refinePhase(self, x, f_basis, sg, lr_param, epoch_param):
        # Refine a self-consistent phase in a desired space group, given a sequence function
        # x should be the initial guesses of the given space group - No f(s)!!! - It contains xi, psi, cell, *angle

        # Set up the optimized x values
        # We want to remove cell and stress from x so no differentiation graph is calculated for them. Especially important if we are using V * incompress as a loss
        x_opt = {"xi": x["xi"], "psi": x["psi"]}
        cell = x["cell"]
        angle = x["angle"] if "angle" in x else jnp.array([])

        ### Strictly enforcing self-consistent equations
        # This method uses gradient descent on the cell and angle parameters
        print(
            "Optimization: refining structure with gradient descent on cell param using stress"
        )
        # Specify a linear schedule for field variables
        lr = optax.linear_schedule(
            init_value=lr_param["field_init_value"],
            end_value=lr_param["field_end_value"],
            transition_steps=lr_param["transition"],
        )
        # Specify a linear schedule for cell variables - gradient descent alpha
        cell_lr = jnp.linspace(
            lr_param["cell_init_value"],
            lr_param["cell_end_value"],
            lr_param["transition"],
        )
        # Then append constant end value to reach num_epochs_refine
        remaining_steps = epoch_param["epochs"] - lr_param["transition"]
        cell_lr = jnp.concatenate(
            [cell_lr, lr_param["cell_end_value"] * jnp.ones(remaining_steps)]
        )

        # Set up the optimizer
        optimizer = optax.adam(learning_rate=lr, nesterov=True)
        opt_state = optimizer.init(x_opt)

        # Unpack required space group, u0_list
        space_group = list(self.space_group[sg].data.values())
        u0 = list(self.space_group[sg].u0.values())

        early_stopping = EarlyStopping(threshold=1e-5)
        grace_period = epoch_param["grace"]  # Don't check for x epochs

        # The step to take for the optimizer. Use scftRefine which shifts psi and xi. Good for when we are near an exact answer
        @jax.jit
        def step(
            opt_state,
            x,
            cell,
            angle,
            f_basis,
            space_group=space_group,
            u0_list=u0,
            dt=self.dt,
            Ns=self.Ns,
            stencil=self.stencil,
            basis=self.basis,
        ):
            output, grads = jax.value_and_grad(scftRefine, has_aux=True)(
                x, cell, angle, f_basis, space_group, u0_list, dt, Ns, stencil, basis
            )
            loss_value, aux = output

            # Update the SCFT fields - not unit cell parameters
            updates, opt_state = optimizer.update(grads, opt_state, params=x)
            x = optax.apply_updates(x, updates)

            return opt_state, x, loss_value, aux

        for epoch in range(epoch_param["epochs"]):
            # Compute the loss and gradients
            opt_state, x_opt, loss_value, aux = step(
                opt_state, x_opt, cell, angle, f_basis
            )
            (
                w_matlab,
                w_report,
                incompress_report,
                cell_stress,
                free_energy,
                free_energy_disorder,
            ) = aux

            if epoch + 1 > grace_period:
                # Manually update cell with stress
                cell = cell - cell_lr[epoch] * cell_stress

            if (epoch + 1) % epoch_param["print"] == 0:
                print(
                    f"Epoch {epoch + 1}, Loss: {loss_value:.6e}, Free Energy: {free_energy:.3f}, Incompressible Loss: {incompress_report:.6e}, W Loss: {w_report:.6e}, W MATLAB Criteria: {w_matlab:.6e}, Stress Loss: {cell_stress}, Cell: {cell}"
                )

            if loss_value <= 1e-12:
                print(
                    f"Early stopping due to below tolerance, last epoch: Epoch {epoch + 1}, Loss: {loss_value:.6e}, Free Energy: {free_energy:.3f}, Incompressible MAE: {incompress_report:.6e}, W MAE: {w_report:.6e}, Stress Loss: {cell_stress}, Cell: {cell}"
                )
                break

            if (early_stopping.check_stop(jnp.array(free_energy))) and (
                epoch >= grace_period
            ):
                print(
                    f"Early stopping due to energy difference in window below threshold, last epoch: Epoch {epoch + 1}, Loss: {loss_value:.6e}, Free Energy: {free_energy:.3f}, Incompressible Loss: {incompress_report:.6e}, W Loss: {w_report:.6e}, Stress Loss: {cell_stress}, Cell: {cell}"
                )
                break

        # Update the stored converged phases - We can only freely shift the xi field. Do so because in scftRefine this occurs
        if angle.size > 0:
            # shift the xi field
            xi = self.shiftXi(x_opt["xi"], sg)
            x = {"xi": xi, "psi": x_opt["psi"], "cell": cell, "angle": angle}
        else:
            xi = self.shiftXi(x_opt["xi"], sg)
            x = {"xi": xi, "psi": x_opt["psi"], "cell": cell}

        return x

    def run_scft(
        self, sg: str, load: bool = False, load_dict: dict = {}, save: bool = False
    ):
        # Perform a default run of the SCFT calculations for a given space group sg.
        # Step 1 : Generate f_basis. Assumes that sequence has f_params already set
        f_basis = self.sequence.transform_fun(self.sequence.f_params)

        # Step 2: Generate an initial guess for the space group sg or load (depending on the load flag)
        if load:
            load_file = load_dict["filename"]
            load_cell = load_dict["cell"]
            load_angle = load_dict.get("angle")  # None if no angle
            sg_list = sg if isinstance(sg, list) else [sg]
            x_dict, sg_dict = self.loadPhase(load_file, sg_list, load_cell, load_angle)
        else:
            initial_star_dict = {
                sg: self.space_group[sg].initial_star
            }  # This is a dictionary, just for consistency
            x_dict, sg_dict = self.initialGuess(
                f_basis, [sg], initial_star_dict, 0.02
            )  # 0.1

        # Step 3: For the 2 initial guesses, run the SCFT calculations
        total_epochs = 10000
        # Find the dimension of the space group to determine the learning rate and cell initialization values
        if self.space_group[sg].unit_cell == "lam":
            dim = 1
        elif (
            self.space_group[sg].unit_cell == "hex"
            or self.space_group[sg].unit_cell == "square"
            or self.space_group[sg].unit_cell == "rectangular"
            or self.space_group[sg].unit_cell == "oblique"
        ):
            dim = 2
        else:
            dim = 3

        if dim < 3:
            field_init_value = 1e-2  # 1e-3
            cell_init_value = 5e-1  # 1e-1
        else:
            field_init_value = 1e-2  # 1e-2
            cell_init_value = 5e-1  # 5e-1
        lr_param = {
            "lr_label": "normal",
            "field_init_value": field_init_value,
            "cell_init_value": cell_init_value,
            "field_end_value": 1e-3,  # field_init_value
            "cell_end_value": 1e-1,  # 5e-2
            "transition": total_epochs,
            "mix_transition": total_epochs,
        }
        epoch_param = {
            "first_optimization": 200,  # 200
            "second_optimization": 200,  # 200
            "third_optimization": total_epochs,
            "grace": 200,
            "print": 2000,
        }
        loss_param = {"scft_weight": 1, "free_energy_weight": 0, "energy_dif_weight": 1}

        x_sg = {}  # This will store the converged structures
        p_xg = {}  # This will store the evaluated parameters for each structure

        for struct_name, sg in sg_dict.items():
            print(f"Converging {struct_name} structure in {sg} space group")
            x_temp = self.convergePhase(
                x_dict[struct_name],
                f_basis,
                sg,
                F_target=0,
                lr_param=lr_param,
                epoch_param=epoch_param,
                loss_param=loss_param,
            )  # Choice of F target is very important

            print(f"Optimization complete, writing into index {struct_name}")
            x_sg[struct_name] = x_temp

            # Evaluate the phase
            x_eval = {"f_basis": f_basis} | x_sg[struct_name]
            p_xg[struct_name] = self.evaluatePhase(x_eval, sg)

            # If the save flag is True, save the results
            if save:
                save_filename = f"scft_{self.sequence.name}_{struct_name}"
                SaveResults().save_scft_results(
                    x_sg[struct_name], p_xg[struct_name], save_filename
                )
                SaveResults().save_matlab(
                    x_sg[struct_name],
                    p_xg[struct_name],
                    f_basis,
                    save_filename,
                )

        return x_sg, p_xg
