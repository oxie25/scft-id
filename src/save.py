# Class for performing manipulations on a converged result and saving the outputs
# Contains methods for both the result, as well as trajectories
import numpy as np
import jax.numpy as jnp
from scipy.io import savemat, loadmat
import pickle

from .utils import CONVERGED_STRUCTURES_DIR, STRUCTURES_DIR
from .structures_utils import RealField


class SaveResults:
    def __init__(self):
        pass

    def save_scft_results(self, x: dict, p: dict, save_name: str):
        # This method takes the parameter output of the SCFT and saves it as .npy files
        filename = CONVERGED_STRUCTURES_DIR / f"{save_name}_psi.npy"
        np.save(filename, x["psi"])

        filename = CONVERGED_STRUCTURES_DIR / f"{save_name}_phi.npy"
        np.save(filename, p["phi"])

    def save_matlab(self, x: dict, p: dict, f_final: jnp.ndarray, save_name: str):
        # This method save the results in a format suitable for MATLAB
        psi = x["psi"]
        xi = x["xi"]
        w = p["w"]
        Q = p["Q"]
        L_ss = p["L_ss"]
        phi = p["phi"]
        cell = x["cell"]
        angle = x["angle"] if "angle" in x else jnp.array([])

        filename = CONVERGED_STRUCTURES_DIR / f"{save_name}.mat"
        # Save all required data into a single .mat file
        savemat(
            filename,
            {
                "f_final": np.array(f_final),
                "L_ss": np.array(L_ss),
                "psi": np.array(psi),
                "xi": np.array(xi),
                "w": np.array(w),
                "Q": np.array(Q),
                "cell": np.array(cell),
                "phi": np.array(phi),
                "angle": np.array(angle) if angle.size > 0 else np.array([]),
            },
        )

    def save_scattering(self, sg: str, geo: list, x: dict, p: dict, save_name: str):
        # This method saves the scattering data as a .csv file
        phi = p["phi"]

        tauIdx_file = STRUCTURES_DIR / f"{sg}_tauIdx.mat"
        h2ijk_file = STRUCTURES_DIR / f"{sg}_h2ijk.mat"

        tauIdx_mat = loadmat(tauIdx_file)
        h2ijk_mat = loadmat(h2ijk_file)

        tauIdx = tauIdx_mat["tauidx"][0]
        h2ijk = h2ijk_mat["h2ijk"]

        Nwaves = np.shape(h2ijk)[0]

        # Initialize class
        TargetIsoSurface = RealField(phi, geo)

        # Generate coordinate grid
        TargetIsoSurface.generate_coordinate_grid()

        # Generate geometry
        TargetIsoSurface.generate_geometry()

        # Convert field to 3D
        field_1d, F = TargetIsoSurface.field_to_cartesian(
            tauIdx, h2ijk, Nwaves, matlab=True
        )

        # Save field_1d as a csv file
        filename = STRUCTURES_DIR / f"{save_name}_phi_rf.csv"
        np.savetxt(
            filename,
            field_1d,
            delimiter=",",
        )

    def save_id(self, data, save_name):
        # This method dumps the data into a pickle file for inverse design
        filename = CONVERGED_STRUCTURES_DIR / f"{save_name}.pickle"
        with open(filename, "wb") as f:
            pickle.dump(data, f)


class TrajectoryRecorder:
    epoch: list
    energy: list
    energy_disorder: list
    loss: list
    incompress_loss: list
    w_loss: list
    phi_dev_loss: list
    sequence: list
    phi_hat: list
    cell: list
    w_matlab_loss: list
    write_size: int
    Ns: int
    Ndim: int

    def __init__(self, write_size, Ns, Ndim):
        self.epoch = [None] * write_size
        self.energy = [None] * write_size
        self.energy_disorder = [None] * write_size
        self.loss = [None] * write_size
        self.incompress_loss = [None] * write_size
        self.w_loss = [None] * write_size
        self.phi_dev_loss = [None] * write_size
        self.sequence = [None] * write_size
        self.phi_hat = [None] * write_size
        self.cell = [None] * write_size
        self.w_matlab_loss = [None] * write_size
        self.write_size = write_size
        self.Ns = Ns
        self.Ndim = Ndim
        self.current_index = 0

    def record(
        self,
        epoch,
        free_energy,
        free_energy_disorder,
        loss_value,
        incompress_report,
        w_report,
        w_matlab,
        phi_report,
        f_fun,
        phi_hat,
        cell,
    ):
        if self.current_index >= self.write_size:
            raise IndexError("Preallocated list size exceeded.")
        self.epoch[self.current_index] = epoch
        self.energy[self.current_index] = free_energy
        self.energy_disorder[self.current_index] = free_energy_disorder
        self.loss[self.current_index] = loss_value
        self.incompress_loss[self.current_index] = incompress_report
        self.w_loss[self.current_index] = w_report
        self.w_matlab_loss[self.current_index] = w_matlab
        self.phi_dev_loss[self.current_index] = phi_report
        self.sequence[self.current_index] = f_fun
        self.phi_hat[self.current_index] = phi_hat.flatten()
        self.cell[self.current_index] = cell.flatten()
        self.current_index += 1

    @property
    def trajectory(self):
        return {
            "energy": self.energy,
            "energy_disorder": self.energy_disorder,
            "loss": self.loss,
            "incompress_loss": self.incompress_loss,
            "w_loss": self.w_loss,
            "phi_dev_loss": self.phi_dev_loss,
            "sequence": self.sequence,
            "phi_hat": self.phi_hat,
            "cell": self.cell,
            "w_matlab_loss": self.w_matlab_loss,
        }


class NSTrajectoryRecorder:
    epoch: list
    target_loss: list
    target_refining_loss: list
    alternate_loss: list
    target_energy: list
    disorder_energy: list
    alternate_energy: list
    sequence: list
    write_size: int
    Ns: int
    Ndim: int

    def __init__(self, write_size, Ns, Ndim):
        self.epoch = [None] * write_size
        self.target_loss = [None] * write_size
        self.target_refining_loss = [None] * write_size
        self.alternate_loss = [None] * write_size
        self.target_energy = [None] * write_size
        self.disorder_energy = [None] * write_size
        self.alternate_energy = [None] * write_size
        self.sequence = [None] * write_size
        self.write_size = write_size
        self.Ns = Ns
        self.Ndim = Ndim
        self.current_index = 0

    def record(
        self,
        epoch,
        target_loss,
        target_refining_loss,
        alternate_loss,
        target_energy,
        disorder_energy,
        alternate_energy,
        sequence,
    ):
        if self.current_index >= self.write_size:
            raise IndexError("Preallocated list size exceeded.")
        self.epoch[self.current_index] = epoch
        self.target_loss[self.current_index] = target_loss
        self.target_refining_loss[self.current_index] = target_refining_loss
        self.alternate_loss[self.current_index] = alternate_loss
        self.target_energy[self.current_index] = target_energy
        self.disorder_energy[self.current_index] = disorder_energy
        self.alternate_energy[self.current_index] = alternate_energy
        self.sequence[self.current_index] = sequence
        self.current_index += 1

    @property
    def trajectory(self):
        return {
            "target_loss": self.target_loss,
            "target_refining_loss": self.target_refining_loss,
            "alternate_loss": self.alternate_loss,
            "target_energy": self.target_energy,
            "disorder_energy": self.disorder_energy,
            "alternate_energy": self.alternate_energy,
            "sequence": self.sequence,
        }
