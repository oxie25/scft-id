import os
import gc
import jax.numpy as jnp
import equinox as eqx
import pandas as pd
from pathlib import Path

# Import your own structure tools
from . import structures_utils


class SpaceGroup(eqx.Module):
    name: str
    data: dict  # Holds FT, FS, kbz, Gwt, dRIndex, cellIndex, angleIndex, angleArray arrays
    u0: dict  # {'u_fwd': jnp.array, 'u_rev': jnp.array}
    unit_cell: str
    cell: jnp.ndarray  # Unit cell parameters
    angle: jnp.ndarray  # Angle parameters
    initial_star: list  # Initial star of the space group

    @staticmethod
    def from_data(name, data, unit_cell, cell, angle, initial_star, Ns):
        # Calculate additional cell geometry properties
        cell_geometry = structures_utils.R_prop_calc(unit_cell)
        data["dRIndex"] = cell_geometry[0]
        data["cellIndex"] = cell_geometry[1]
        data["angleIndex"] = cell_geometry[2]
        data["angleArray"] = cell_geometry[3]

        # Generate u0
        FT = data["FT"]
        Ntaus = FT.shape[1]
        u0 = {}
        u0["u_fwd"] = jnp.ones(
            (Ntaus,)
        )  # jnp.zeros((Ntaus, Ns)).at[:, 0].set(jnp.ones((Ntaus,)))
        u0["u_rev"] = jnp.ones(
            (Ntaus,)
        )  # jnp.zeros((Ntaus, Ns)).at[:, 0].set(jnp.ones((Ntaus,)))

        return SpaceGroup(
            name=name,
            data=data,
            unit_cell=unit_cell,
            u0=u0,
            cell=cell,
            angle=angle,
            initial_star=initial_star,
        )


class SpaceGroupCollection(dict):
    @staticmethod
    def load_data(folder_path, prefixes, descriptors):
        space_group_data = {}
        for prefix in prefixes:
            space_group_data[prefix] = {}
            file_name = f"{prefix}.npz"
            file_path = os.path.join(folder_path, file_name)
            if os.path.exists(file_path):
                print(f"Loading {file_name}...")
                jax_data = jnp.load(file_path, allow_pickle=True, mmap_mode="r")
                for desc in descriptors:
                    space_group_data[prefix][desc] = jax_data[desc]
            else:
                raise FileNotFoundError(f"File not found: {file_name}")
            gc.collect()
        return space_group_data

    @classmethod
    def from_files(
        cls,
        desired_sg: list,
        sg_info_file: Path,
        structure_folder: Path,
        Ns=101,
        descriptors=None,
    ):
        if descriptors is None:
            descriptors = ["FT", "FS", "kbz", "Gwt"]

        df_sg_info = pd.read_csv(sg_info_file, index_col=0)

        unit_cell = df_sg_info["Unit Cell"].loc[desired_sg].to_dict()
        cell_param = {
            sg: jnp.array(
                [float(x) for x in str(val).split(", ")]
                if isinstance(val, str)
                else [float(val)]
            )
            for sg, val in df_sg_info["Cell"].loc[desired_sg].items()
        }
        angle_param = {
            sg: (
                jnp.array([float(x) for x in str(val).split(", ")])
                if isinstance(val, str) and val.strip() != ""
                else jnp.array([])
            )
            for sg, val in df_sg_info["Angle"].loc[desired_sg].items()
        }
        initial_star = {
            sg: [int(x) for x in str(val).split(",")]
            for sg, val in df_sg_info["Initial Star"].loc[desired_sg].items()
        }

        space_group_raw_data = cls.load_data(structure_folder, desired_sg, descriptors)

        groups = {}
        for sg_name in desired_sg:
            groups[sg_name] = SpaceGroup.from_data(
                sg_name,
                space_group_raw_data[sg_name],
                unit_cell[sg_name],
                cell_param[sg_name],
                angle_param[sg_name],
                initial_star[sg_name],
                Ns,
            )

        return cls(groups)
