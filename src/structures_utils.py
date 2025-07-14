# Functions and classes for handling space group structures and fields

# Author: Oliver Xie, 2025

import gc
import os

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision


def create_target_phi(
    target_generation_flag, target_sg, target_file, unit_cell, space_group, n=2
):
    # n is the number of non-cancelled wavevectors
    ### Generate the target phi field
    if target_sg in unit_cell.keys():
        print(f"The desired {target_sg} space group is in the list")
    else:
        print(f"The desired {target_sg} space group is not in list. Exiting with error")
        return False

    if target_generation_flag == "load":
        target_file = f"./converged_structures/{target_file}"
        phi_target = jnp.load(target_file, allow_pickle=True, mmap_mode="r")
        print(f"The file {target_file} successfully loaded")
        return phi_target

    elif target_generation_flag == "generate":
        # Currently not working
        print("Generating target phi field is currently not implemented")


def R_prop_calc(cell):
    # Return dRIndex, cellIndex, angleIndex, angleArray
    # dRIndex chooses which of the derivative calculations to use for each unit cell. Must be specified in the same order as the supplied cell parameters
    # cellIndex maps the regressed cell parameter array into a common cell length array of [L1, L2, L3]
    # angleIndex maps the regressed angle parameter array (if applicable) into a common cell angle array of [alpha, beta, gamma], if no angles are regressible, leave as empty list
    # angleArray supplies the common cell array for each unit cell, because some have predefined non pi/2 angles

    if cell == "lam":
        # Parameter order is L1
        dRIndex = jnp.array([0])  # L1
        cellIndex = [jnp.array([0])]
        angleIndex = []
        angleArray = jnp.array([jnp.pi / 2, jnp.pi / 2, jnp.pi / 2])  # Fixed

    elif cell == "oblique":
        # Parameter order is L1, L2, gamma
        dRIndex = jnp.array([0, 1, 7])  # L1, L2, gamma
        cellIndex = [jnp.array([0]), jnp.array([1])]
        angleIndex = [jnp.array([2])]
        angleArray = jnp.array(
            [jnp.pi / 2, jnp.pi / 2, jnp.pi / 2]
        )  # Variable in gamma - Just for initialization purposes

    elif cell == "square":
        # Parameter order is L1
        dRIndex = jnp.array([4])  # L1=L2
        cellIndex = [jnp.array([0, 1])]
        angleIndex = []
        angleArray = jnp.array([jnp.pi / 2, jnp.pi / 2, jnp.pi / 2])  # Fixed

    elif cell == "rectangular":
        # Parameter order is L1, L2
        dRIndex = jnp.array([0, 1])  # L1, L2
        cellIndex = [jnp.array([0]), jnp.array([1])]
        angleIndex = []
        angleArray = jnp.array([jnp.pi / 2, jnp.pi / 2, jnp.pi / 2])  # Fixed

    elif cell == "hex":
        # Parameter order is L1
        dRIndex = jnp.array([4])  # L1=L2
        cellIndex = [jnp.array([0, 1])]
        angleIndex = []
        angleArray = jnp.array([jnp.pi / 2, jnp.pi / 2, 2 * jnp.pi / 3])  # Fixed

    elif cell == "triclinic":
        # Parameter order is L1, L2, L3, alpha, beta, gamma
        dRIndex = jnp.array([0, 1, 2, 5, 6, 7])  # All parameters
        cellIndex = [jnp.array([0]), jnp.array([1]), jnp.array([2])]
        angleIndex = [jnp.array([0]), jnp.array([1]), jnp.array([2])]
        angleArray = jnp.array(
            [jnp.pi / 2, jnp.pi / 2, jnp.pi / 2]
        )  # Variable - Just for initialization purposes

    elif cell == "monoclinic":
        # Parameter order is L1, L2, L3, beta
        dRIndex = jnp.array([0, 1, 2, 6])  # L1, L2, L3, beta
        cellIndex = [jnp.array([0]), jnp.array([1]), jnp.array([2])]
        angleIndex = [jnp.array([1])]
        angleArray = jnp.array(
            [jnp.pi / 2, jnp.pi / 2, jnp.pi / 2]
        )  # Variable in beta - Just for initialization purposes

    elif cell == "rhombohedral":
        # Parameter order is L1, alpha
        dRIndex = jnp.array([3, 8])  # L1=L2=L3 and alpha=beta=gamma
        cellIndex = [jnp.array([0, 1, 2])]
        angleIndex = [jnp.array([0])]
        angleArray = jnp.array(
            [jnp.pi / 2, jnp.pi / 2, jnp.pi / 2]
        )  # Variable - Just for initialization purposes

    elif cell == "hexagonal":
        # Parameter order is L1, L2
        dRIndex = jnp.array([4, 2])
        cellIndex = [jnp.array([0, 1]), jnp.array([2])]
        angleIndex = []
        angleArray = jnp.array([jnp.pi / 2, jnp.pi / 2, 2 * jnp.pi / 3])  # Fixed

    elif cell == "orthorhombic":
        # Parameter order is L1, L2, L3
        dRIndex = jnp.array([0, 1, 2])
        cellIndex = [jnp.array([0]), jnp.array([1]), jnp.array([2])]
        angleIndex = []
        angleArray = jnp.array([jnp.pi / 2, jnp.pi / 2, jnp.pi / 2])  # Fixed

    elif cell == "tetragonal":
        # Parameter order is L1, L3
        dRIndex = jnp.array([4, 2])  # L1=L2, L3
        cellIndex = [jnp.array([0, 1]), jnp.array([2])]
        angleIndex = []
        angleArray = jnp.array([jnp.pi / 2, jnp.pi / 2, jnp.pi / 2])  # Fixed

    elif cell == "cubic":
        # Parameter order is L1
        dRIndex = jnp.array([3])  # L1=L2=L3
        cellIndex = [jnp.array([0, 1, 2])]
        angleIndex = []
        angleArray = jnp.array([jnp.pi / 2, jnp.pi / 2, jnp.pi / 2])  # Fixed

    return dRIndex, cellIndex, angleIndex, angleArray


# Function to load files
def load_data(folder_path, prefixes, descriptors):
    space_group_data = {}
    for prefix in prefixes:
        space_group_data[prefix] = {}
        for desc in descriptors:
            file_name = f"{prefix}_{desc}.npy"
            file_path = os.path.join(folder_path, file_name)

            if os.path.exists(file_path):
                print(f"Loading {file_name}...")

                jax_data = jnp.load(file_path, allow_pickle=True, mmap_mode="r")

                # Convert to JAX array
                # row.append(jax_data)
                space_group_data[prefix][desc] = jax_data
            else:
                raise FileNotFoundError(f"File not found: {file_name}")

        # master_list.append(row)
        # del row
        gc.collect()

    return space_group_data


def generate_space_group(space_group_data, unit_cell):
    for prefix, data in space_group_data.items():
        cell_geometry = R_prop_calc(unit_cell[prefix])
        # Append additional values (dRIndex, cellIndex, angleIndex, angleArray) as new columns
        data["dRIndex"] = cell_geometry[0]
        data["cellIndex"] = cell_geometry[1]
        data["angleIndex"] = cell_geometry[2]
        data["angleArray"] = cell_geometry[3]

    return space_group_data


def generate_u0(space_group, Ns):
    u0_dict = {}
    # Generate a dictionary where each key is a space group and the values are dictionaries with keys 'u_fwd' and 'u_rev'
    for key, value in space_group.items():
        FT = value["FT"]  # Extract FT from the space group data
        Ntaus = FT.shape[1]  # number of columns is Ntaus
        u0 = jnp.ones(shape=(Ntaus,))
        u_initial_fwd = jnp.zeros((Ntaus, Ns))
        u_initial_fwd = u_initial_fwd.at[:, 0].set(u0)
        u_initial_rev = jnp.zeros((Ntaus, Ns))
        u_initial_rev = u_initial_rev.at[:, 0].set(u0)

        u0_dict[key] = {"u_fwd": u_initial_fwd, "u_rev": u_initial_rev}

    return u0_dict


# Change maxval to be the max seen in the initial structure
def field_initialize(Ntaus, Ns):
    # Initialization of structures in alternate space groups
    # JIT compile
    rng_key = jax.random.key(0)
    # Initialize a field guess
    # For xi field, can be negative
    field_shape = (Ntaus, Ns)  # Shape of the w field grid over time
    field_init = 0.01 * jax.random.normal(rng_key, shape=field_shape)  # Make this small

    return field_init


def field_star_initialize(
    domain, sign, star_list, Nstars, Ns, FS, FT, g_fun, value=0.1
):
    # Domain and sign controls the guess type
    # Domain: g_normal, g_flip
    # Sign: positive, negative

    # Initialize a field guess
    field_shape = (Nstars, Ns)  # Shape of the w field grid over time
    field_init = jnp.zeros((field_shape))
    field_init = field_init.at[0, :].set(1)  # Set as 1 in zero wavevector

    # Initialize using the supplied star list
    star_values = jnp.zeros(shape=(Nstars,))

    # value = 0.1 # Can change - but 0.1 works well
    star_values = star_values.at[0].set(1.0)  # Incompressibility condition
    for star in star_list:
        if sign == "positive":
            star_values = star_values.at[star].set(1 * value)
        elif sign == "negative":
            star_values = star_values.at[star].set(-1 * value)
        else:
            print(f"Sign flag of {sign} is not recognized")

    # Only fill in non-zero wavevectors
    guess_vector = jnp.expand_dims(star_values, 1)

    if domain == "g_normal":
        field_init_a = field_init
        field_init_b = field_init
        field_init_a = (
            field_init_a.at[:, :].set(jnp.tile(guess_vector, (1, Ns))) * g_fun
        )
        field_init_b = field_init_b.at[:, :].set(jnp.tile(-guess_vector, (1, Ns))) * (
            1.0 - g_fun
        )
        field_init = field_init_a + field_init_b
    elif domain == "g_flip":
        field_init_a = field_init
        field_init_b = field_init
        field_init_a = field_init_a.at[:, :].set(jnp.tile(guess_vector, (1, Ns))) * (
            1.0 - g_fun
        )
        field_init_b = (
            field_init_b.at[:, :].set(jnp.tile(-guess_vector, (1, Ns))) * g_fun
        )
        field_init = field_init_a + field_init_b
        # display(g_fun)
    else:
        print(f"Domain flag of {domain} is not recognized")

    field_init = field_init.at[0, :].set(
        1
    )  # Set as 1 in zero wavevector to enforce incompressibility

    field_tau = jnp.absolute(
        jnp.matmul(FS, field_init)
    )  # Remove possible negative densities

    # Pass back and forth between the two spaces to ensure Gnorm is correct
    field_star = jnp.matmul(FT, field_tau)
    field_tau = jnp.matmul(FS, field_star)

    return field_tau


class RealField:
    def __init__(self, field, p):
        # Initialize with the field, the length of each dimension L, the number of points in each dimension 'N', the dimension 'dim', the unit cell 'cell'
        L, N, dim, cell = p

        self.field = field
        self.L = L
        self.N = N
        self.dim = dim
        self.cell = cell

    def generate_coordinate_grid(self):
        # Generate the fractional crystallographic coordinate grid
        Ni, Nj, Nk = self.N

        # If dimension is less than 3, have 2 pts in non-simulated dimension (for plotting on 3D)
        if self.dim == 1:
            self.i_pts = np.linspace(0, 1, Ni)
            self.j_pts = np.linspace(0, 1, 48)
            self.k_pts = np.linspace(0, 1, 48)
            self.Ni = Ni
            self.Nj = 48
            self.Nk = 48
        elif self.dim == 2:
            self.i_pts = np.linspace(0, 1, Ni)
            self.j_pts = np.linspace(0, 1, Nj)
            self.k_pts = np.linspace(0, 1, 48)
            self.Ni = Ni
            self.Nj = Nj
            self.Nk = 48
        elif self.dim == 3:
            self.i_pts = np.linspace(0, 1, Ni)
            self.j_pts = np.linspace(0, 1, Nj)
            self.k_pts = np.linspace(0, 1, Nk)
            self.Ni = Ni
            self.Nj = Nj
            self.Nk = Nk

        # Generate meshgrid
        self.I, self.J, self.K = np.mgrid[
            0 : 1 : self.Ni * 1j, 0 : 1 : self.Nj * 1j, 0 : 1 : self.Nk * 1j
        ]

    def generate_geometry(self):
        # Generate the necessary geometry for the problem depending on the unit cell
        # Angles are given as alpha, beta, and gamma following crystallographic convention
        if self.cell == "lam":
            self.Lx = self.L[0]
            self.Ly = 1
            self.Lz = 1
            self.angle = [np.pi / 2, np.pi / 2, np.pi / 2]
        elif self.cell == "square":
            self.Lx = self.L[0]
            self.Ly = self.L[1]
            self.Lz = 1  # Default to 1 for extra dimension
            self.angle = [np.pi / 2, np.pi / 2, np.pi / 2]
        elif self.cell == "hex":
            self.Lx = self.L[0]
            self.Ly = self.L[1]
            self.Lz = 1  # Default to 1 for extra dimension
            self.angle = [np.pi / 2, np.pi / 2, 2 * np.pi / 3]
        elif self.cell == "cubic":
            self.Lx = self.L[0]
            self.Ly = self.L[1]
            self.Lz = self.L[2]
            self.angle = [np.pi / 2, np.pi / 2, np.pi / 2]
        elif self.cell == "tetragonal":
            self.Lx = self.L[0]
            self.Ly = self.L[1]
            self.Lz = self.L[2]
            self.angle = [np.pi / 2, np.pi / 2, np.pi / 2]
        elif self.cell == "hexagonal":
            self.Lx = self.L[0]
            self.Ly = self.L[1]
            self.Lz = self.L[2]
            self.angle = [np.pi / 2, np.pi / 2, 2 * np.pi / 3]

    def generate_cartesian_grid(self):
        # Generate the cartesian grid for plotting
        # The generated grid spacing is fractional crystallographic coordinates, convert to Cartesian
        alpha, beta, gamma = self.angle

        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)

        a = self.Lx
        b = self.Ly
        c = self.Lz

        # Calculate the volume factor
        V = (
            a
            * b
            * c
            * np.sqrt(
                1
                - cos_alpha**2
                - cos_beta**2
                - cos_gamma**2
                + 2 * cos_alpha * cos_beta * cos_gamma
            )
        )

        # Transformation matrix from fractional to Cartesian coordinates
        M = np.round(
            np.array(
                [
                    [a, b * cos_gamma, c * cos_beta],
                    [
                        0,
                        b * sin_gamma,
                        c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
                    ],
                    [0, 0, V / (a * b * sin_gamma)],
                ]
            ),
            6,
        )

        # Apply the transformation to the meshgrid of fractional coordinates
        # Stack the I, J, K grids and apply the transformation to each point
        cart_X, cart_Y, cart_Z = np.dot(
            M, [self.I.flatten(), self.J.flatten(), self.K.flatten()]
        )

        # Reshape back to meshgrid shape
        X = cart_X.reshape((self.Ni, self.Nj, self.Nk))
        Y = cart_Y.reshape((self.Ni, self.Nj, self.Nk))
        Z = cart_Z.reshape((self.Ni, self.Nj, self.Nk))

        # Extract x, y, z as separate meshgrids
        self.X = X
        self.Y = Y
        self.Z = Z

        return self.X, self.Y, self.Z, M

    def field_to_cartesian(self, tauIdx, h2ijk, N, matlab):
        # Transform the field to a full cartesian coordinate (Nx, Ny, Nz)
        # The structure of tauIdx is that it is an array of arrays which contain the index of the full vector to assign the tau field
        if matlab:
            # Remember to subtract 1 for matlab notation if the source is matlab
            subtract = 1
        else:
            subtract = 0

        field_tau = self.field

        # Generate a 'full vector'
        field_real = np.zeros((N,))

        for idx, tau_map in enumerate(tauIdx):
            field_real[tau_map - subtract] = field_tau[idx]

        # Convert from full vector to matrix
        F = np.zeros((self.Ni, self.Nj, self.Nk))

        # Place the values from y into the 3D matrix at the specified coordinates
        for idx, (i, j, k) in enumerate(h2ijk):
            F[i - subtract, j - subtract, k - subtract] = field_real[idx]

        # Expand the field if 1D or 2D into 3D
        if self.dim == 1:
            F[:, 1, 0] = F[:, 0, 0]  # Translate in y
            F[:, 0, 1] = F[:, 0, 0]  # Translate in z
            F[:, 1, 1] = F[:, 0, 0]  # Translate in both y and z
        if self.dim == 2:
            F_flat = F[:, :, 0]
            F = np.repeat(F_flat[:, :, np.newaxis], self.Nk, axis=2)
            # F = np.tile(F_flat, (1, 1, self.Nk))

        self.F = F.T  # Need to transpose to use with mgrid

        F = self.F

        return field_real, F
