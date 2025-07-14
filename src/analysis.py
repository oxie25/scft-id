### Class for loading in a pickle file and performing analysis on it
### Supported operations:
#       plot any structure with pyvista
#       plot any sequence and interaction matrix
#       plot the trajectories of inverse design and negative selection
#       plot the block fractions and chi as a functio of steps
#       plot the free energies of the final refined structures and alternates


import pickle
import numpy as np
import jax
import jax.numpy as jnp
import os
import sys
from dataclasses import dataclass
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.sequence import Sequence
from src.utils import (
    STRUCTURES_DIR,
    PROJECT_ROOT,
    FIGURES_DIR,
    GLOBAL_NS
)
from dataclasses import field

jax.config.update("jax_enable_x64", True)  # True for double precision
jax.config.update("jax_platform_name", "cpu")  # Use only CPU for data analysis


@dataclass
class Analysis:
    id_x: dict = field(default_factory=dict)
    id_trajectory: dict = field(default_factory=dict)
    negative_x: dict = field(default_factory=dict)
    negative_trajectory: dict = field(default_factory=dict)
    negative_alt_x: dict = field(default_factory=dict)
    negative_refine_x: dict = field(default_factory=dict)
    negative_refine_alt_x: dict = field(default_factory=dict)
    x_refine: dict = field(default_factory=dict)
    x_alt_dict: dict = field(default_factory=dict)
    data_keys: list = field(default_factory=list)
    ns_iterations: int = 0

    def __init__(self, file, block_type, basis, Nbasis):
        # Load the pickle file and assign the data to different parts
        if file.endswith(".pickle"):
            with open(file, "rb") as f:
                data = pickle.load(f)

            # Read the keys
            if not isinstance(data, dict):
                raise ValueError(
                    "Data must be a dictionary loaded from the pickle file."
                )

            # The keys by default should be
            self.data_keys = list(data.keys())

            # Assign all the data to the class attributes
            self.id_x = data.get("id_x", {})
            self.id_trajectory = data.get("id_trajectory", {})
            self.negative_x = data.get("negative_x", {})
            self.negative_trajectory = data.get("negative_trajectory", {})
            self.negative_alt_x = data.get("negative_alt_x", {})
            self.negative_refine_x = data.get("negative_refine_x", {})
            self.negative_refine_alt_x = data.get("negative_refine_alt_x", {})
            self.x_refine = data.get("x_refine", {})
            self.x_alt_dict = data.get("x_alt_dict", {})

            # Find the maximum number of negative selection iterations used
            self.ns_iterations = len(
                [
                    item
                    for item in self.negative_trajectory.get("target_energy", [])
                    if item is not None
                ]
            )
        else:
            print("Provided file is not a pickle file. Only valid method is plotPhi")

        # Get plotting parameters
        self.font = "Arial"  # Default font for plotting
        self.font_size = 14  # Default font size for plotting

        # The number of points in the sequence
        self.N = GLOBAL_NS
        self.Nbasis = Nbasis  # Number of basis functions
        self.basis = basis  # The type of basis used for the sequence
        self.block_type = block_type  # The type of block transformation used
        self.s = jnp.linspace(0, 1, self.N)

        # Evaluate the energy given a set of x and a space group
        # Format x for evaluation. If x contains a f_basis, use it, otherwise use f_basis from the input
        f_basis = self.id_x.get(
            "f_basis", None
        )  # Ensure f_basis is set in x if not already present

        # Transform this f_basis into a sequence
        if block_type == "multiblock":
            self.Ndim = f_basis.shape[0] - 1  # Number of dimensions in the basis
            self.Np = f_basis.shape[1]  # Number of points in the sequence
        elif block_type == "multiblock_taper":
            self.Ndim = f_basis.shape[0] - 2
            self.Np = f_basis.shape[1]  # Number of points in the sequence
        elif block_type == "no_transform" and self.basis == "chebyshev_basis":
            self.Ndim = f_basis.shape[1]
            self.Np = f_basis.shape[0]

    def evaluate(self, x, sg: str, value: str, f_basis=None):
        from src.scft import SCFT
        from src.sequence import Sequence
        from src.space_group import SpaceGroupCollection

        # Evaluate the energy given a set of x and a space group
        # Format x for evaluation. If x contains a f_basis, use it, otherwise use f_basis from the input
        f_basis = x.get(
            "f_basis", f_basis
        )  # Ensure f_basis is set in x if not already present

        sequenceClass = Sequence.generate(
            name="test",
            basis=self.basis,
            transform=self.block_type,
            Nbasis=self.Nbasis,
            Np=self.Np,
            Ndim=self.Ndim,
            f_params=f_basis,
        )
        f_param = sequenceClass.transform_fun(f_basis)

        x_eval = x.copy()
        x_eval["f_basis"] = f_param  # Add the f_basis to the x

        # Set up SCFT
        space_groups_to_load = [sg]
        sg_info_file = PROJECT_ROOT / "space_group_info.csv"

        # Set up the space group geometries
        sg_collection = SpaceGroupCollection.from_files(
            desired_sg=space_groups_to_load,
            sg_info_file=sg_info_file,
            structure_folder=STRUCTURES_DIR,
        )
        scft = SCFT(space_group=sg_collection, sequence=sequenceClass)

        # Evaluate the phase
        p = scft.evaluatePhase(x_eval, sg)

        return p[value]  # Value should be 'energy', 'phi', 'w_matlab'

    def determineEqm(self, sg_dict: dict):
        # We plot the energy after x_refine and x_alt_dict
        ns_x = self.negative_x
        x_alt_dict = self.negative_refine_alt_x
        f_basis = ns_x.get("f_basis", None)
        x_refine = self.negative_refine_x

        target_sg = sg_dict["target"]

        w_error = {}

        free_energy_target = self.evaluate(
            x_refine, f_basis=f_basis, sg=target_sg, value="energy"
        )
        free_energy_disorder = self.evaluate(
            x_refine, f_basis=f_basis, sg=target_sg, value="disorder energy"
        )
        w_error["target"] = self.evaluate(
            x_refine, f_basis=f_basis, sg=target_sg, value="w_matlab"
        )

        # For every alternate structure, we evaluate the free energy
        free_energy_alt = {}
        for key, x in x_alt_dict.items():
            sg_alt = sg_dict[key]
            free_energy_alt[key] = self.evaluate(
                x, f_basis=f_basis, sg=sg_alt, value="energy"
            )
            w_error[key] = self.evaluate(
                x, f_basis=f_basis, sg=sg_alt, value="w_matlab"
            )

        free_energies_dif = {
            key: value - free_energy_target for key, value in free_energy_alt.items()
        }
        free_energies_dif["disorder"] = free_energy_disorder - free_energy_target

        # Go through each free_energies_dif and determine if any are below zero. If any are, this is not an equilibrium
        flag = True
        for key, value in free_energies_dif.items():
            if value < 0:
                print(
                    f"Structure {key} is below the target energy. This is not an equilibrium structure."
                )
                flag = False

        if flag:
            print(
                "All structures are above the target energy. This is an equilibrium structure."
            )

        return flag, free_energies_dif, w_error, free_energy_target

    def plotTrajectory(
        self,
        file_name: str,
        alternates: list,
        ylim_1: tuple = None,
        ylim_2: tuple = None,
    ):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # Ensure required data exists
        if not self.id_trajectory or not self.negative_trajectory:
            raise ValueError("Trajectory data is missing or incomplete.")

        # Extract relevant data from the class attributes
        id_energy = self.id_trajectory.get("energy", [])
        id_disorder_energy = self.id_trajectory.get("energy_disorder", [])
        ns_energy = self.negative_trajectory.get("target_energy", [])
        ns_disorder_energy = self.negative_trajectory.get("disorder_energy", [])
        ns_alternate_energy = self.negative_trajectory.get("alternate_energy", [])

        # Ensure alternate energy data is available
        if not ns_alternate_energy:
            raise ValueError("Alternate energy data is missing or incomplete.")

        # Extract alternate energies for specific structures
        ns_alternate_energy = np.array(
            ns_alternate_energy[: self.ns_iterations]
        )  # Convert to a numpy array

        # Use Arial font
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.size"] = 16

        # Generate a viridis color palette
        viridis_palette = [plt.cm.viridis(i / 10) for i in range(11)]

        # Plot the energy and disorder energy
        fig = plt.figure(figsize=(12, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.0)

        # Left plot: full-scale first optimization
        ax0 = plt.subplot(gs[0])
        ax0.plot(id_energy, label="Energy", color=viridis_palette[0], linestyle="-")
        ax0.plot(
            id_disorder_energy, label="Disorder Energy", color="black", linestyle=":"
        )
        ax0.spines["right"].set_visible(True)
        ax0.tick_params(labelright=False)

        # Right plot: append the NS energies
        ax1 = plt.subplot(gs[1], sharey=ax0)
        ax1.plot(
            ns_disorder_energy, label="NS Disorder Energy", color="black", linestyle=":"
        )

        # Plot alternate energies based on the alternates list
        for i, alt_idx in enumerate(alternates):
            if alt_idx < ns_alternate_energy.shape[1]:
                color_idx = min(i, 10)  # Ensure we don't exceed viridis palette
                ax1.plot(
                    ns_alternate_energy[:, alt_idx],
                    label=f"alternate_{alt_idx}",
                    color=viridis_palette[color_idx],
                    alpha=0.7,
                    linestyle="-",
                )

        ax1.plot(ns_energy, label="NS Energy", color=viridis_palette[0], linestyle="-")
        ax1.spines["left"].set_visible(True)
        ax1.tick_params(labelleft=False)
        ax1.tick_params(axis="y", which="both", left=False, right=False)

        # Restrict y-axis
        if ylim_1:
            ax0.set_ylim(ylim_1[0], ylim_1[1])
            ax1.set_ylim(ylim_1[0], ylim_1[1])

        # Save the figure
        filepath = FIGURES_DIR / f"{file_name}_F_trajectory.png"
        plt.savefig(filepath, dpi=1200, bbox_inches="tight", transparent=False)
        plt.show()

        # Plot the cell parameter evolution in the initial design
        cell_id_trajectory = np.array(
            self.id_trajectory.get("cell", [])
        )  # Ensure the data is converted to a numpy array
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.size"] = 16

        viridis_palette = [plt.cm.viridis(i / 10) for i in range(11)]

        plt.figure(figsize=(4, 2))

        # Plot each column of cell_id_trajectory as a separate line
        for i in range(cell_id_trajectory.shape[1]):
            plt.plot(
                cell_id_trajectory[:, i],
                color=viridis_palette[2 * i],
                linestyle="-",
                linewidth=2,
            )

        print(f"Final cell parameters: {cell_id_trajectory[-1, :]}")

        filepath = FIGURES_DIR / f"{file_name}_id_cell.png"
        plt.savefig(filepath, dpi=600, bbox_inches="tight", transparent=True)
        plt.show()

        # Plot the delta F in the trajectory for the negative selection
        fig = plt.figure(figsize=(4, 2))

        # Calculate the delta F for the negative selection against the target
        max_ns_iterations = self.ns_iterations
        delta_F_disorder = np.array(ns_disorder_energy[:max_ns_iterations]) - np.array(
            ns_energy[:max_ns_iterations]
        )

        viridis_palette = [plt.cm.viridis(i / 10) for i in range(11)]

        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.size"] = 16

        plt.plot(
            delta_F_disorder,
            label="NS Disorder Energy",
            color="black",
            linestyle="--",
            linewidth=2,
        )

        # Plot delta F for alternates based on the alternates list
        for i, alt_idx in enumerate(alternates):
            if alt_idx < ns_alternate_energy.shape[1]:
                delta_F_alt = np.array(
                    ns_alternate_energy[:max_ns_iterations, alt_idx]
                ) - np.array(ns_energy[:max_ns_iterations])
                color_idx = min(i, 10)  # Ensure we don't exceed viridis palette
                plt.plot(
                    delta_F_alt,
                    label=f"alternate_{alt_idx}",
                    color=viridis_palette[color_idx],
                    linestyle="-",
                    linewidth=2,
                )

        plt.hlines(0, xmin=0, xmax=self.ns_iterations - 1, color="black", linestyle=":")
        plt.xlim([0, self.ns_iterations - 1])
        plt.xlim([0, max_ns_iterations - 1])
        if ylim_2:
            plt.ylim(ylim_2[0], ylim_2[1])
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins=2, integer=True))
        plt.gca().yaxis.set_minor_locator(plt.NullLocator())
        plt.gca().yaxis.set_ticks([plt.gca().get_ylim()[0], 0, plt.gca().get_ylim()[1]])

        # Remove legend from the plot and add it outside
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        filepath = FIGURES_DIR / f"{file_name}_ns_delta_F.png"
        plt.savefig(filepath, dpi=600, bbox_inches="tight", transparent=True)
        plt.show()

    def plotSequence(
        self, file_name: str, iteration: dict = None, figure_size: tuple = (2.5, 2.5)
    ):
        import matplotlib.pyplot as plt
        from src.scft_utils import pairwise, sequence_to_identity

        # Use iteration to specify the exact sequence to plot. If none, plot the first (initial guess), result after ID, and the result after NS
        sequenceClass = Sequence.generate(
            name="test",
            basis=self.basis,
            transform=self.block_type,
            Nbasis=self.Nbasis,
            Np=self.Np,
            Ndim=self.Ndim,
        )
        basis_fun = sequenceClass.basis_fun(
            self.s, self.Nbasis
        )  # This stores the matrix basis rather than the function. We never need the function itself

        if iteration is None:
            # Get the initial and after inverse design sequence
            sequence = self.id_trajectory["sequence"]

            initial_sequence = sequence[0]
            initial_sequence = jnp.dot(basis_fun.T, initial_sequence)
            initial_lss = pairwise(initial_sequence)
            initial_g = sequence_to_identity(initial_sequence)
            # Get the final sequence after inverse design
            final_id_sequence = sequence[-1]
            final_id_sequence = jnp.dot(basis_fun.T, final_id_sequence)
            final_id_lss = pairwise(final_id_sequence)
            final_id_g = sequence_to_identity(final_id_sequence)

            # Get the final sequence after negative selection. Unfortunately due to choice in the code, trajectory is already transformed into a sequence here
            ns_sequence = self.negative_trajectory["sequence"]
            final_ns_sequence = ns_sequence[self.ns_iterations - 1]
            # final_ns_sequence = jnp.dot(basis_fun.T, final_ns_sequence) - Not needed in current code
            final_ns_lss = pairwise(final_ns_sequence)
            final_ns_g = sequence_to_identity(final_ns_sequence)

            # Plot all the sequences
            plt.rcParams["font.family"] = self.font
            plt.rcParams["font.size"] = self.font_size
            fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
            x_axis = np.linspace(0, 1, len(initial_sequence))
            ax[0].plot(
                x_axis,
                initial_sequence,
                marker="o",
                label="Initial Sequence",
                color=plt.cm.viridis(0.2),
            )
            ax[0].plot(
                x_axis,
                initial_g,
                label="Initial Identity",
                color="black",
                linestyle="--",
            )
            ax[0].set_title("Initial Sequence")
            ax[0].set_xlabel("Normalized Sequence Position")
            ax[0].set_ylabel("Sequence Value")
            ax[0].set_xticks(np.linspace(0, 1, 5))
            ax[0].set_xticklabels([f"{x:.2f}" for x in np.linspace(0, 1, 5)])

            ax[1].plot(
                x_axis,
                final_id_sequence,
                marker="o",
                label="Inverse Design Sequence",
                color=plt.cm.viridis(0.4),
            )
            ax[1].plot(
                x_axis,
                final_id_g,
                label="Inverse Design Identity",
                color="black",
                linestyle="--",
            )
            ax[1].set_title("Inverse Design Sequence")
            ax[1].set_xlabel("Normalized Sequence Position")
            ax[1].set_ylabel("Sequence Value")
            ax[1].set_xticks(np.linspace(0, 1, 5))
            ax[1].set_xticklabels([f"{x:.2f}" for x in np.linspace(0, 1, 5)])

            ax[2].plot(
                x_axis,
                final_ns_sequence,
                marker="o",
                label="Negative Selection Sequence",
                color=plt.cm.viridis(0.6),
            )
            ax[2].plot(
                x_axis,
                final_ns_g,
                label="Negative Selection Identity",
                color="black",
                linestyle="--",
            )
            ax[2].set_title("Negative Selection Sequence")
            ax[2].set_xlabel("Normalized Sequence Position")
            ax[2].set_ylabel("Sequence Value")
            ax[2].set_xticks(np.linspace(0, 1, 5))
            ax[2].set_xticklabels([f"{x:.2f}" for x in np.linspace(0, 1, 5)])

            plt.tight_layout()
            # Save the figure
            filepath = FIGURES_DIR / f"{file_name}_sequence.png"
            plt.savefig(filepath, dpi=600, bbox_inches="tight", transparent=True)
            plt.show()

            # Plot heatmap for the negative selection interaction matrix for the three calculated lss
            fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))

            # Determine the maximum value across all LSS heatmaps
            vmax = max(initial_lss.max(), final_id_lss.max(), final_ns_lss.max())

            # Initial LSS heatmap
            ax[0].imshow(initial_lss, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
            ax[0].set_title("Initial LSS", fontsize=14, fontname="Arial")
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            for spine in ax[0].spines.values():
                spine.set_visible(False)

            # Final ID LSS heatmap
            ax[1].imshow(final_id_lss, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
            ax[1].set_title("Inverse Design LSS", fontsize=14, fontname="Arial")
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            for spine in ax[1].spines.values():
                spine.set_visible(False)

            # Final NS LSS heatmap
            ax[2].imshow(final_ns_lss, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
            ax[2].set_title("Negative Selection LSS", fontsize=14, fontname="Arial")
            ax[2].set_xticks([])
            ax[2].set_yticks([])
            for spine in ax[2].spines.values():
                spine.set_visible(False)

            # Add a colorbar to the right of the subplots
            im = ax[2].imshow(
                final_id_lss, cmap="viridis", aspect="auto", vmin=0, vmax=vmax
            )  # Define 'im' here
            cbar = fig.colorbar(
                im, ax=ax[2], orientation="vertical", fraction=0.05, pad=0.02
            )

            # Adjust x-axis to be normalized (0 to 1) for heatmaps
            for i, sequence in enumerate(
                [initial_sequence, final_id_sequence, final_ns_sequence]
            ):
                x_axis = np.linspace(0, 1, len(sequence))
                ax[i].set_xticks(np.linspace(0, len(sequence) - 1, 5))
                ax[i].set_xticklabels([f"{x:.2f}" for x in np.linspace(0, 1, 5)])
            # Adjust heatmap axes to normalized (0 to 1) scale
            for i, lss in enumerate([initial_lss, final_id_lss, final_ns_lss]):
                ax[i].set_xticks(np.linspace(0, lss.shape[1] - 1, 5))
                ax[i].set_xticklabels([f"{x:.2f}" for x in np.linspace(0, 1, 5)])
                ax[i].set_yticks(np.linspace(0, lss.shape[0] - 1, 5))
                ax[i].set_yticklabels([f"{y:.2f}" for y in np.linspace(0, 1, 5)])

            plt.tight_layout()
            # Save the figure
            filepath = FIGURES_DIR / f"{file_name}_lambda.png"
            plt.savefig(filepath, dpi=600, bbox_inches="tight", transparent=True)
            plt.show()

            # Create a separate plot for just the final NS LSS
            plt.figure(figsize=figure_size)
            plt.imshow(final_ns_lss, cmap="cmo.dense", aspect="equal")

            # Set normalized axes (0 to 1)
            plt.xticks([0, final_ns_lss.shape[1] - 1], ["0", "1"])
            plt.yticks([0, final_ns_lss.shape[0] - 1], ["0", "1"])

            # Add colorbar
            cbar = plt.colorbar(orientation="vertical", fraction=0.05, pad=0.2)

            plt.tight_layout()
            # Save the figure
            filepath = FIGURES_DIR / f"{file_name}_final_ns_lambda.png"
            plt.savefig(filepath, dpi=600, bbox_inches="tight", transparent=True)
            plt.show()

        else:
            # Get the selected sequence and LSS based on the provided iteration
            stage = iteration.get("phase", None)
            sequence_index = iteration.get("iteration", 0)
            if stage == "negative_selection":
                selected_sequence = self.negative_trajectory["sequence"][sequence_index]
                selected_sequence = jnp.dot(basis_fun.T, selected_sequence)
                selected_lss = pairwise(selected_sequence)
            elif stage == "inverse_design":
                selected_sequence = self.id_trajectory["sequence"][sequence_index]
                selected_sequence = jnp.dot(basis_fun.T, selected_sequence)
                selected_lss = pairwise(selected_sequence)
            else:
                raise ValueError(
                    "Invalid phase specified. Use 'negative_selection' or 'inverse_design'."
                )

            ## Plot the sequence
            fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))

            # Sequence plot
            plt.rcParams["font.family"] = self.font
            plt.rcParams["font.size"] = self.font_size
            ax[0].plot(selected_sequence, marker="o", color=plt.cm.viridis(0.4))
            ax[0].set_title(f"Sequence (Iteration {sequence_index})")
            ax[0].set_xlabel("Sequence Discretized Point")
            ax[0].set_ylabel("Sequence Value")

            # LSS heatmap
            im = ax[1].imshow(
                selected_lss, cmap="viridis", aspect="auto", vmin=0, vmax=40
            )
            ax[1].set_title(f"LSS (Iteration {sequence_index})")
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            for spine in ax[1].spines.values():
                spine.set_visible(False)

            # Add a colorbar for the LSS heatmap
            cbar = fig.colorbar(
                im, ax=ax[1], orientation="vertical", fraction=0.05, pad=0.02
            )
            cbar.ax.tick_params(labelsize=12)

            plt.tight_layout()
            # Save the figure
            filepath = FIGURES_DIR / f"{file_name}_iteration_{sequence_index}.png"
            plt.savefig(filepath, dpi=600, bbox_inches="tight", transparent=True)
            plt.show()

    def plotBlockChi(self, stage: str, file_name: str):
        ### Note: this only works for block copolymer - do not pass a taper in
        # For a sequence, identify the blocks, and plot the block fractions and the χ matrix as a heatmap. This only works for a known f_basis. We can only do this for the result of id and the result of ns
        # stage indicates if it is ns or id
        import matplotlib.pyplot as plt
        import seaborn as sns

        from src.sequence import Sequence
        from src.sequence_utils import calculate_fractions
        from src.scft_utils import pairwise

        # Extract the f_basis from x_id and x_ns
        f_basis_id = self.id_x.get("f_basis", None)
        f_basis_ns = self.negative_x.get("f_basis", None)

        # The first row of f_basis is the block fractions after passing through calculate_fractions
        block_frac_id = np.array(calculate_fractions(f_basis_id[0, :]))
        block_frac_ns = np.array(calculate_fractions(f_basis_ns[0, :]))

        # Find the sequence
        sequenceClass = Sequence.generate(
            name="test",
            basis=self.basis,
            transform=self.block_type,
            Nbasis=self.Nbasis,
            Np=self.Np,
            Ndim=self.Ndim,
            f_params=f_basis_id,
        )

        # This only works for block copolymers, no other changes needed
        seq_id = sequenceClass.transform_fun(f_basis_id)
        seq_ns = sequenceClass.transform_fun(f_basis_ns)

        lss_id = np.array(pairwise(seq_id))
        lss_ns = np.array(pairwise(seq_ns))

        # If stage is 'id', we use the id fractions and chi matrix, otherwise we use the ns fractions and chi matrix
        if stage == "id":
            fractions = block_frac_id
            # Number of blocks
            n_blocks = len(fractions)

            # Generate the pairwise combinations
            block_index, _ = self._find_indices_for_block(fractions)
            pairwise_combinations = self._generate_pairwise_combinations(n_blocks)

            # block_index is a dictionary. Now for each pair in pairwise_combinations, we can find the indices for each block, Use this to sample from lss to find the appropriate chi
            chi_matrix = np.zeros((n_blocks, n_blocks))
            for pair in pairwise_combinations:
                block1, block2 = pair
                index_block1 = block_index[block1]
                index_block2 = block_index[block2]
                sampled_values = lss_id[index_block1, index_block2]
                chi_matrix[block2, block1] = (
                    sampled_values  # Fill only the lower diagonal
                )

            filepath = FIGURES_DIR / f"{file_name}_id_block_chi.png"

        elif stage == "ns":
            fractions = block_frac_ns
            # Number of blocks
            n_blocks = len(fractions)

            # Generate the pairwise combinations
            block_index, _ = self._find_indices_for_block(fractions)
            pairwise_combinations = self._generate_pairwise_combinations(n_blocks)

            # block_index is a dictionary. Now for each pair in pairwise_combinations, we can find the indices for each block, Use this to sample from lss to find the appropriate chi
            chi_matrix = np.zeros((n_blocks, n_blocks))
            for pair in pairwise_combinations:
                block1, block2 = pair
                index_block1 = block_index[block1]
                index_block2 = block_index[block2]
                sampled_values = lss_ns[index_block1, index_block2]
                chi_matrix[block2, block1] = (
                    sampled_values  # Fill only the lower diagonal
                )

            filepath = FIGURES_DIR / f"{file_name}_ns_block_chi.png"
        else:
            raise ValueError("Stage must be either 'id' or 'ns'.")

        # Plotting
        plt.rcParams["font.family"] = self.font
        plt.rcParams["font.size"] = self.font_size

        fig, ax = plt.subplots(
            1, 2, figsize=(10, 6), gridspec_kw={"width_ratios": [1, 4]}
        )

        # Bar plot for fractions
        # ax[0].barh(range(n_blocks), fractions, color='black')
        ax[0].set_yticks(range(n_blocks))
        ax[0].set_yticklabels([f"B{i}" for i in range(n_blocks)])
        ax[0].invert_yaxis()
        ax[0].set_xlabel("Fraction")
        bars = ax[0].barh(range(n_blocks), fractions, color="black")
        for i, (bar, fraction) in enumerate(zip(bars, fractions)):
            ax[0].text(
                fraction / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{fraction:.3f}",
                va="center",
                ha="center",
                fontsize=12,
                color="white",
            )

        # Half matrix heatmap
        mask = np.triu(np.ones_like(chi_matrix, dtype=bool))
        sns.heatmap(
            chi_matrix,
            mask=mask,
            cmap="Blues",
            ax=ax[1],
            cbar_kws={"label": "χ"},
            annot=True,
            fmt=".2f",
            annot_kws={"size": self.font_size, "color": "black"},
        )
        ax[1].set_xticks([x + 0.5 for x in range(n_blocks)])
        ax[1].set_xticklabels(
            [f"B{i}" for i in range(n_blocks)], rotation=45, ha="center"
        )
        ax[1].set_yticks([])  # Remove y ticks

        plt.tight_layout()
        plt.savefig(filepath, dpi=1200, bbox_inches="tight", transparent=True)
        plt.show()

    def plotEnergy(self, file_name: str, block_type: str, stage: str, sg_dict: dict):
        import matplotlib.pyplot as plt

        target_sg = sg_dict[
            "target"
        ]  # The target space group is the one we are interested in

        if stage == "id":
            # We plot the energy after x_refine and x_alt_dict
            id_x = self.id_x
            x_alt_dict = self.x_alt_dict
            f_basis = id_x.get("f_basis", None)
            x_refine = self.x_refine

            free_energy_target = self.evaluate(
                x_refine, f_basis=f_basis, sg=target_sg, value="energy"
            )
            free_energy_disorder = self.evaluate(
                x_refine, f_basis=f_basis, sg=target_sg, value="disorder energy"
            )

            # For every alternate structure, we evaluate the free energy
            free_energy_alt = {}
            for key, x in x_alt_dict.items():
                sg_alt = sg_dict[key]
                free_energy_alt[key] = self.evaluate(
                    x, f_basis=f_basis, sg=sg_alt, value="energy"
                )

            filepath = FIGURES_DIR / f"{file_name}_id_energy.png"
        elif stage == "ns":
            # We plot the energy after x_refine and x_alt_dict
            ns_x = self.negative_x
            x_alt_dict = self.negative_refine_alt_x
            f_basis = ns_x.get("f_basis", None)
            x_refine = self.negative_refine_x

            free_energy_target = self.evaluate(
                x_refine, f_basis=f_basis, sg=target_sg, value="energy"
            )
            free_energy_disorder = self.evaluate(
                x_refine, f_basis=f_basis, sg=target_sg, value="disorder energy"
            )

            # For every alternate structure, we evaluate the free energy
            free_energy_alt = {}
            for key, x in x_alt_dict.items():
                sg_alt = sg_dict[key]
                free_energy_alt[key] = self.evaluate(
                    x, f_basis=f_basis, sg=sg_alt, value="energy"
                )

            filepath = FIGURES_DIR / f"{file_name}_ns_energy.png"
        else:
            raise ValueError("Stage must be either 'id' or 'ns'.")

        plt.rcParams["font.family"] = self.font
        plt.rcParams["font.size"] = self.font_size

        # Create a categorical plot
        fig, ax = plt.subplots(figsize=(2, 3.6))

        # Plot horizontal lines for each value in the DataFrame
        free_energies_dif = {
            key: value - free_energy_target for key, value in free_energy_alt.items()
        }
        x_position = 1  # Single x-coordinate for all hlines

        # Only plot the free energies for the structures we are interested in
        free_energies_dif = {
            key: value
            for key, value in free_energies_dif.items()
            if key in sg_dict.keys()
        }

        # Generate distinct shades of viridis based on the number of structures to consider
        num_shades = len(sg_dict.keys()) + 1
        shades_of_viridis = [
            plt.cm.viridis(i / (num_shades - 1)) for i in range(num_shades)
        ]

        for i, (key, value) in enumerate(free_energies_dif.items()):
            ax.hlines(
                value,
                xmin=x_position - 0.4,
                xmax=x_position + 0.4,
                colors=shades_of_viridis[i % len(shades_of_viridis)],
                label=key,
                linewidth=2,
            )

        # Plot the disorder energy as a dotted horizontal line
        ax.hlines(
            free_energy_disorder - free_energy_target,
            xmin=x_position - 0.4,
            xmax=x_position + 0.4,
            colors="black",
            linestyle="--",
            label="Disorder Energy",
            linewidth=2,
        )

        # Set x-axis ticks and labels
        ax.set_xticks([x_position])
        ax.set_xticklabels([])
        # Add a legend to the plot
        ax.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10, title="Structures"
        )
        plt.tight_layout()
        plt.savefig(filepath, dpi=1200, bbox_inches="tight", transparent=True)
        plt.show()

    def plotStructures(
        self,
        file_name: str,
        sg_dict: dict,
        stage: str,
        method: str,
        block=None,
        cutoff: float = 0.5,
        cmin: float = 0.0,
        cmax: float = 1.0,
        origin: tuple = (0, 0, 0.5),
        normal: tuple = (0, 0, 1),
    ):
        # Generate a volume plot of a single structure using pyvista
        # sg_dict is a dictionary where the keys are the names of the structure and the values are the space group. If the target is the desired structure, the key should be 'target'
        # block_type is either 'multiblock' or 'multiblock_taper'
        # stage indicates if it is ns or id
        # block inidicates which block to plot. If None, plot the identity block
        import pandas as pd
        from scipy.io import loadmat
        from src.plotting_utils import IsosurfacePlotter
        from src.sequence_utils import calculate_fractions
        from src.scft_utils import integral, sequence_to_identity
        import pyvista as pv

        pv.global_theme.allow_empty_mesh = True


        # Extract the necessary data
        psi_dict = {}
        cell_dict = {}
        if stage == "id":
            f_basis = self.id_x.get("f_basis", None)
            # Get the required space group. Loop through sg_dict
            for key, sg in sg_dict.items():
                if key == "target":
                    psi_dict[key] = self.x_refine.get("psi", None)
                    cell_dict[key] = self.x_refine.get("cell", None)
                else:
                    # look in the alternate structures
                    psi_dict[key] = self.x_alt_dict.get(f"{key}", {}).get("psi", None)
                    cell_dict[key] = self.x_alt_dict.get(f"{key}", {}).get("cell", None)

        elif stage == "ns":
            f_basis = self.negative_x.get("f_basis", None)
            # Get the required space group. Loop through sg_dict
            for key, sg in sg_dict.items():
                if key == "target":
                    psi_dict[key] = self.x_refine.get("psi", None)
                    cell_dict[key] = self.x_refine.get("cell", None)
                else:
                    # look in the alternate structures
                    psi_dict[key] = self.x_alt_dict.get(f"{key}", {}).get("psi", None)
                    cell_dict[key] = self.x_alt_dict.get(f"{key}", {}).get("cell", None)
        else:
            raise ValueError("Stage must be either 'id' or 'ns'.")

        # The first row of f_basis is the block fractions after passing through calculate_fractions
        block_frac = np.array(
            calculate_fractions(f_basis[0, :])
        )  # Need to make this numpy
        _, block_indices_dict = self._find_indices_for_block(block_frac)

        sequenceClass = Sequence.generate(
            name="test",
            basis=self.basis,
            transform=self.block_type,
            Nbasis=self.Nbasis,
            Np=self.Np,
            Ndim=self.Ndim,
        )

        # Generate the sequence. This is to find the identity function
        basis_fun = sequenceClass.basis_fun(
            self.s, self.Nbasis
        )  # This stores the matrix basis rather than the function. We never need the function itself
        sequence = sequenceClass.transform_fun(f_basis)
        sequence = jnp.dot(basis_fun.T, sequence)

        # Set up the space group geometries
        sg_info_file = PROJECT_ROOT / "space_group_info.csv"

        # For every space group in sg_dict, we need to load the structure and generate the isosurface
        for key, sg in sg_dict.items():
            # Load the necessary structure files for plotting
            tauIdx_file = Path(STRUCTURES_DIR) / f"{sg}_tauIdx.mat"
            h2ijk_file = Path(STRUCTURES_DIR) / f"{sg}_h2ijk.mat"

            tauIdx_mat = loadmat(tauIdx_file)
            h2ijk_mat = loadmat(h2ijk_file)

            tauIdx = tauIdx_mat["tauidx"][0]
            h2ijk = h2ijk_mat["h2ijk"]

            # From the file with space group informaton, we can extract the discretization
            df_sg_info = pd.read_csv(sg_info_file, index_col=0)

            unit_cell = df_sg_info["Unit Cell"].loc[sg]
            Nx = df_sg_info["Nx"].loc[sg]
            Ny = df_sg_info["Ny"].loc[sg]
            Nz = df_sg_info["Nz"].loc[sg]

            # Get required parameters for the isosurface plotter
            N = [Nx, Ny, Nz]
            cell, dim = self._cell_fill(
                unit_cell, cell_dict[key]
            )  # Fill the cell based on the unit cell type
            # the param input is cell lengths, discretization, dimension, and unit cell
            param = cell, N, dim, unit_cell

            # Calculate the phi for plotting
            if block is None:
                # If no block is specified, we calculate the phi for the identity block
                g_fun = sequence_to_identity(sequence)
                phi_hat = np.array(
                    integral(psi_dict[key] * g_fun, sequenceClass.stencil)
                )  # Make into a numpy array
            elif isinstance(block, int):
                # If a specific block is requested, we need to calculate the phi for that block
                # We can use the block indices. g_fun is zero everywhere except between the block indices
                start_stop = block_indices_dict[
                    block
                ]  # This is a tuple of (start, stop) indices for the block
                g_fun = jnp.zeros((self.N,))  # Initialize g_fun as a zero array
                g_fun = g_fun.at[start_stop[0] : start_stop[1]].set(1.0)
                phi_hat = np.array(
                    integral(psi_dict[key] * g_fun, sequenceClass.stencil)
                )  # Make into a numpy array
            elif isinstance(block, tuple):
                # If block is a tuple, use the tuple values as start and stop indices
                start, stop = block
                g_fun = jnp.zeros((self.N,))  # Initialize g_fun as a zero array
                g_fun = g_fun.at[start:stop].set(1.0)
                phi_hat = np.array(
                    integral(psi_dict[key] * g_fun, sequenceClass.stencil)
                )  # Make into a numpy array
            else:
                raise ValueError(
                    "Block must be None, an integer, or a tuple of (start, stop) indices"
                )

            # Find total number of waves
            Nwaves = np.shape(h2ijk)[0]

            # This is the plotting code
            # Initialize class
            TargetIsoSurface = IsosurfacePlotter(phi_hat, param)

            # Generate coordinate grid
            TargetIsoSurface.generate_coordinate_grid()

            # Generate geometry
            TargetIsoSurface.generate_geometry()

            # Generate Cartesian grid
            X, Y, Z, M = TargetIsoSurface.generate_cartesian_grid()

            # Convert field to 3D
            F = TargetIsoSurface.field_to_cartesian(tauIdx, h2ijk, Nwaves, matlab=True)

            # Prepare plotter
            plotter = pv.Plotter(off_screen=bool(file_name))
            plotter.enable_anti_aliasing("msaa")

            if method == "volume":
                # Use the isovolume method
                TargetIsoSurface.isovolume(
                    plotter=plotter, min_clim=cmin, max_clim=cmax
                )

                if file_name:
                    if block is not None:
                        screenshot_path = (
                            FIGURES_DIR / f"{file_name}_{key}_block_{block}.png"
                        )
                    else:
                        screenshot_path = (
                            FIGURES_DIR / f"{file_name}_{key}_identity.png"
                        )
                    plotter.screenshot(
                        screenshot_path, return_img=False, window_size=[1920, 1080]
                    )
                else:
                    plotter.show()
            elif method == "surface":
                # Use the filled method
                # Prepare plotter
                plotter = pv.Plotter(off_screen=bool(file_name))
                plotter.enable_anti_aliasing("msaa")

                TargetIsoSurface.filled(cutoff=cutoff, plotter=plotter)

                if file_name:
                    if block is not None:
                        screenshot_path = (
                            FIGURES_DIR / f"{file_name}_{key}_block_{block}_filled.png"
                        )
                    else:
                        screenshot_path = (
                            FIGURES_DIR / f"{file_name}_{key}_identity_filled.png"
                        )
                    plotter.screenshot(
                        screenshot_path, return_img=False, window_size=[1920, 1080]
                    )
                else:
                    plotter.show()
            elif method == "slice":
                # Slice the structure
                TargetIsoSurface.slice2d(plotter=plotter, origin=origin, normal=normal)

                if file_name:
                    if block is not None:
                        screenshot_path = (
                            FIGURES_DIR / f"{file_name}_{key}_block_{block}_slice.png"
                        )
                    else:
                        screenshot_path = (
                            FIGURES_DIR / f"{file_name}_{key}_identity_slice.png"
                        )
                    plotter.screenshot(
                        screenshot_path, return_img=False, window_size=[1920, 1080]
                    )
                else:
                    plotter.show()

    def plotPhi(
        self,
        file_name: str,
        phi_file: str,
        cell: list,
        sg: str,
        method: str,
        cutoff: float = 0.5,
        cmin: float = 0.0,
        cmax: float = 1.0,
        origin: tuple = (0, 0, 0.5),
        normal: tuple = (0, 0, 1),
    ):
        # Plots just a phi field (.npy result) using pyvista. Must provide a cell array
        import pandas as pd
        from scipy.io import loadmat
        from src.plotting_utils import IsosurfacePlotter
        import pyvista as pv

        pv.global_theme.allow_empty_mesh = True


        # Set up the space group geometries
        sg_info_file = PROJECT_ROOT / "space_group_info.csv"

        # For every space group in sg_dict, we need to load the structure and generate the isosurface
        # Load the necessary structure files for plotting
        tauIdx_file = Path(STRUCTURES_DIR) / f"{sg}_tauIdx.mat"
        h2ijk_file = Path(STRUCTURES_DIR) / f"{sg}_h2ijk.mat"

        tauIdx_mat = loadmat(tauIdx_file)
        h2ijk_mat = loadmat(h2ijk_file)

        tauIdx = tauIdx_mat["tauidx"][0]
        h2ijk = h2ijk_mat["h2ijk"]

        # From the file with space group informaton, we can extract the discretization
        df_sg_info = pd.read_csv(sg_info_file, index_col=0)

        unit_cell = df_sg_info["Unit Cell"].loc[sg]
        Nx = df_sg_info["Nx"].loc[sg]
        Ny = df_sg_info["Ny"].loc[sg]
        Nz = df_sg_info["Nz"].loc[sg]

        # Get required parameters for the isosurface plotter
        N = [Nx, Ny, Nz]
        cell, dim = self._cell_fill(
            unit_cell, cell
        )  # Fill the cell based on the unit cell type
        # the param input is cell lengths, discretization, dimension, and unit cell
        param = cell, N, dim, unit_cell

        # Load the phi file from the provided path using numpy
        phi_hat = np.load(phi_file)

        # Find total number of waves
        Nwaves = np.shape(h2ijk)[0]

        # This is the plotting code
        # Initialize class
        TargetIsoSurface = IsosurfacePlotter(phi_hat, param)

        # Generate coordinate grid
        TargetIsoSurface.generate_coordinate_grid()

        # Generate geometry
        TargetIsoSurface.generate_geometry()

        # Generate Cartesian grid
        X, Y, Z, M = TargetIsoSurface.generate_cartesian_grid()

        # Convert field to 3D
        F = TargetIsoSurface.field_to_cartesian(tauIdx, h2ijk, Nwaves, matlab=True)

        # Prepare plotter
        plotter = pv.Plotter(off_screen=bool(file_name))
        plotter.enable_anti_aliasing("msaa")

        if method == "volume":
            # Use the isovolume method
            TargetIsoSurface.isovolume(plotter=plotter, min_clim=cmin, max_clim=cmax)

            if file_name:
                screenshot_path = FIGURES_DIR / f"{file_name}_volume.png"
                plotter.screenshot(
                    screenshot_path, return_img=False, window_size=[1920, 1080]
                )
            else:
                plotter.show()
        elif method == "surface":
            # Use the filled method
            # Prepare plotter
            plotter = pv.Plotter(off_screen=bool(file_name))
            plotter.enable_anti_aliasing("msaa")

            TargetIsoSurface.filled(cutoff=cutoff, plotter=plotter)

            if file_name:
                screenshot_path = FIGURES_DIR / f"{file_name}_surface.png"
                plotter.screenshot(
                    screenshot_path, return_img=False, window_size=[1920, 1080]
                )
            else:
                plotter.show()
        elif method == "slice":
            # Slice the structure
            TargetIsoSurface.slice2d(plotter=plotter, origin=origin, normal=normal)

            if file_name:
                screenshot_path = FIGURES_DIR / f"{file_name}_slice.png"
                plotter.screenshot(
                    screenshot_path, return_img=False, window_size=[1920, 1080]
                )
            else:
                plotter.show()

    def movieStructures(
        self,
        file_name: str,
        sg_dict: dict,
        method: str,
        cutoff: float = 0.5,
        cmin: float = 0.0,
        cmax: float = 1.0,
        origin: tuple = (0, 0, 0.5),
        normal: tuple = (0, 0, 1),
        fps=10,
        skip_frames=50,
    ):
        # Generate a movie of the changing inverse designed sequence using pyvista. Only supports the target structure during inverse design
        # sg_dict is a dictionary where the keys are the names of the structure and the values are the space group. If the target is the desired structure, the key should be 'target'
        # method is 'volume', 'surface', or 'slice'
        import pandas as pd
        from scipy.io import loadmat
        from src.plotting_utils import IsosurfacePlotter
        import pyvista as pv
        import imageio

        pv.global_theme.allow_empty_mesh = True

        # Extract the necessary data
        phi_movie_hat = self.id_trajectory.get("phi_hat", None)
        cell_movie = self.id_trajectory.get("cell", None)

        if phi_movie_hat is None or cell_movie is None:
            raise ValueError(
                "Movie data (phi_hat or cell) is missing from id_trajectory."
            )

        # Set up the space group geometries
        sg_info_file = PROJECT_ROOT / "space_group_info.csv"

        sg = sg_dict["target"]
        # Load the necessary structure files for plotting
        tauIdx_file = Path(STRUCTURES_DIR) / f"{sg}_tauIdx.mat"
        h2ijk_file = Path(STRUCTURES_DIR) / f"{sg}_h2ijk.mat"

        tauIdx_mat = loadmat(tauIdx_file)
        h2ijk_mat = loadmat(h2ijk_file)

        tauIdx = tauIdx_mat["tauidx"][0]
        h2ijk = h2ijk_mat["h2ijk"]

        # From the file with space group information, we can extract the discretization
        df_sg_info = pd.read_csv(sg_info_file, index_col=0)

        unit_cell = df_sg_info["Unit Cell"].loc[sg]
        Nx = df_sg_info["Nx"].loc[sg]
        Ny = df_sg_info["Ny"].loc[sg]
        Nz = df_sg_info["Nz"].loc[sg]

        # Get required parameters for the isosurface plotter
        N = [Nx, Ny, Nz]

        # Find total number of waves
        Nwaves = np.shape(h2ijk)[0]

        # Create frame list for movie (sample every few frames to match sequence movie)
        # Start from skip_frames and go to the end with skip_frames intervals
        frame_list = np.arange(skip_frames, np.shape(phi_movie_hat)[0] + skip_frames, skip_frames)

        # Create the plotter for movie generation
        plotter = pv.Plotter(off_screen=True)
        frames = []

        # Process each frame
        for i, frame in enumerate(frame_list):
            if (i + 1) % 10 == 0:
                print(
                    f"Processing frame {frame}/{np.shape(phi_movie_hat)[0] - 1} (Step {i + 1}/{len(frame_list)})"
                )

            # Get cell parameters for this frame and fill based on unit cell type
            cell, dim = self._cell_fill(unit_cell, cell_movie[frame-1])
            param = cell, N, dim, unit_cell

            # Initialize isosurface plotter for this frame
            TargetIsoSurface = IsosurfacePlotter(phi_movie_hat[frame-1], param)

            # Generate coordinate grid
            TargetIsoSurface.generate_coordinate_grid()

            # Generate geometry
            TargetIsoSurface.generate_geometry()

            # Generate Cartesian grid
            _, _, _, _ = TargetIsoSurface.generate_cartesian_grid()

            # Convert field to 3D
            _ = TargetIsoSurface.field_to_cartesian(tauIdx, h2ijk, Nwaves, matlab=True)

            # Clear previous plot
            plotter.clear()

            if method == "volume":
                # Use the isovolume method
                TargetIsoSurface.isovolume(
                    plotter=plotter, min_clim=cmin, max_clim=cmax, suppress=True
                )

            elif method == "surface":
                # Use the filled method
                TargetIsoSurface.filled(cutoff=cutoff, plotter=plotter, suppress=True)

            elif method == "slice":
                # Use the slice method
                TargetIsoSurface.slice2d(
                    plotter=plotter, origin=origin, normal=normal, suppress=True
                )

            # Set camera view
            plotter.view_isometric()

            # Add frame number text
            plotter.add_text(
            f"Step {frame}", position="upper_left", font_size=18, color="black"
            )

            # Capture frame
            img = plotter.screenshot(return_img=True, window_size=[1920, 1080])
            frames.append(img)

        # Save frames as a GIF with consistent timing
        gif_filename = FIGURES_DIR / f"{file_name}_{method}_movie.gif"
        imageio.mimsave(gif_filename, frames, duration=1.0/fps)  # duration in seconds per frame
        print(f"GIF saved as {gif_filename}")

    def movieSequence(
        self, file_name: str, mode: str = "id", fps: int = 10, skip_frames: int = 50
    ):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import matplotlib.cm as cm
        from src.scft_utils import sequence_to_identity

        # Extract sequence trajectory based on mode
        if mode == "id":
            sequence_trajectory = self.id_trajectory.get("sequence", [])
            if not sequence_trajectory:
                raise ValueError("Inverse design sequence trajectory data is missing.")
        elif mode == "ns":
            sequence_trajectory = self.negative_trajectory.get("sequence", [])
            if not sequence_trajectory:
                raise ValueError(
                    "Negative selection sequence trajectory data is missing."
                )
            # Only use up to ns_iterations
            sequence_trajectory = sequence_trajectory[: self.ns_iterations]
        else:
            raise ValueError("Mode must be either 'id' or 'ns'.")

        # Sample frames to reduce file size
        sampled_trajectory = sequence_trajectory[::skip_frames]
        if len(sampled_trajectory) == 0:
            raise ValueError("No frames available after sampling.")

        # Set up plotting parameters
        plt.rcParams["font.family"] = self.font
        plt.rcParams["font.size"] = self.font_size

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 4))

        # Initialize line objects
        Np = sampled_trajectory[0].shape[1]  # Number of dimensions in the sequence
        colormap = cm.get_cmap("viridis", Np + 2)
        sequence_lines = [
            ax.plot([], [], lw=2, color=colormap(i), label=f"Dimension {i+1} : f$_{{{i+1}}}$(s)")[0] for i in range(Np)
        ]
        
        # Create secondary y-axis for identity line
        ax2 = ax.twinx()
        (identity_line,) = ax2.plot([], [], "--", color="black", label="Identity : g(s)")

        # Set plot limits
        x_axis = np.linspace(
            0, 1, sampled_trajectory[0].shape[0]
        )  # The number of rows of any list item of sampled_trajectory
        all_sequences = np.array(sampled_trajectory)
        y_min, y_max = all_sequences.min(), all_sequences.max()
        y_range = y_max - y_min

        ax.set_xlim(0, 1)
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        ax.set_xlabel("Normalized Sequence Position")
        ax.set_ylabel("Sequence Value")
        
        # Set secondary y-axis limits for identity (typically 0 to 1)
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_ylabel("Identity Value")
        
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2)

        # Animation function
        def animate(frame):
            sequence = sampled_trajectory[frame]
            identity = sequence_to_identity(sequence)

            # Handle both 1D and 2D sequences
            if sequence.ndim > 1:
                for j, line in enumerate(sequence_lines):
                    if j < sequence.shape[1]:
                        y = sequence[:, j]
                        line.set_data(x_axis, y)
            else:
                # For 1D sequences, only update the first line
                sequence_lines[0].set_data(x_axis, sequence)

            identity_line.set_data(x_axis, identity)

            # Update title with iteration number
            actual_iteration = frame * skip_frames
            ax.set_title(
                f"Iteration {actual_iteration}"
            )

            # Return all artist objects as a flat list
            return sequence_lines + [identity_line]

        # Create animation
        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=len(sampled_trajectory),
            interval=1000 // fps,  # interval in milliseconds
            blit=True,
            repeat=True,
        )

        # Save as GIF
        gif_path = FIGURES_DIR / f"{file_name}_{mode}_sequence_movie.gif"
        anim.save(gif_path, writer="pillow", fps=fps)

        plt.close(fig)
        print(f"Sequence movie saved as {gif_path}")

    def _generate_pairwise_combinations(self, n_blocks):
        return [(i, j) for i in range(n_blocks) for j in range(i + 1, n_blocks)]

    def _find_indices_for_block(self, block_fraction):
        N = self.N
        cum_frac = np.concatenate([np.array([0]), np.cumsum(block_fraction)])
        block_midpoint_indices = []
        block_start_stop_indices = []
        for i, fraction in enumerate(block_fraction):
            start_idx = cum_frac[i] * N
            end_idx = cum_frac[i + 1] * N
            midpoint = (start_idx + end_idx) / 2
            index = int(np.floor(midpoint))
            block_midpoint_indices.append(index)
            block_start_stop_indices.append(
                (int(np.floor(start_idx)), int(np.floor(end_idx)))
            )

        return {
            block: block_midpoint_indices[block] for block in range(len(block_fraction))
        }, {
            block: block_start_stop_indices[block]
            for block in range(len(block_fraction))
        }

    def _cell_fill(self, unit_cell, cell):
        if unit_cell == "lam":
            dim = 1
            cell = [cell[0], cell[0], cell[0]]
        elif unit_cell == "hex":
            dim = 2
            cell = [cell[0], cell[0], cell[0]]
        elif unit_cell == "square":
            dim = 2
            cell = [cell[0], cell[0], cell[0]]
        elif unit_cell == "rectangular":
            dim = 2
            cell = [cell[0], cell[1], cell[0]]
        elif unit_cell == "cubic":
            dim = 3
            cell = [cell[0], cell[0], cell[0]]
        elif unit_cell == "orthorhombic":
            dim = 3
            cell = [cell[0], cell[1], cell[2]]
        elif unit_cell == "hexagonal":
            dim = 3
            cell = [cell[0], cell[1], cell[1]]
        elif unit_cell == "tetragonal":
            dim = 3
            cell = [cell[0], cell[0], cell[1]]
        else:
            raise ValueError(f"Unknown or unsupported unit cell type: {unit_cell}")

        return cell, dim
