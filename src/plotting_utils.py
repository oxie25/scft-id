# Generates a class
import pyvista as pv
import numpy as np


class IsosurfacePlotter:
    def __init__(self, field, p):
        # Initialize with the field, the length of each dimension L, the number of points in each dimension 'N', the dimension 'dim', the unit cell 'cell'
        L, N, dim, cell = p

        self.field = field
        self.L = L
        self.N = N
        self.dim = dim
        self.cell = cell
        self.cmap = "dense"

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
        elif self.cell == "rectangular":
            self.Lx = self.L[0]
            self.Ly = self.L[1]
            self.Lz = 1  # Default to 1 for extra dimension
            self.angle = [np.pi / 2, np.pi / 2, np.pi / 2]
        elif self.cell == "oblique":
            self.Lx = self.L[0]
            self.Ly = self.L[1]
            self.Lz = 1  # Default to 1 for extra dimension
            self.angle = [np.pi / 2, np.pi / 2, self.L[2]]  # L[2] is gamma angle
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
        elif self.cell == "orthorhombic":
            self.Lx = self.L[0]
            self.Ly = self.L[1]
            self.Lz = self.L[2]
            self.angle = [np.pi / 2, np.pi / 2, np.pi / 2]

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
            F_1D = F[:, 0, 0]
            F = np.repeat(F_1D[:, np.newaxis, np.newaxis], self.Nj, axis=1)
            F = np.repeat(F[:, :, np.newaxis], self.Nk, axis=2)
        if self.dim == 2:
            F_flat = F[:, :, 0]
            F = np.repeat(F_flat[:, :, np.newaxis], self.Nk, axis=2)
            # F = np.tile(F_flat, (1, 1, self.Nk))

        self.F = F.T  # Need to transpose to use with mgrid

        F = self.F

        return F

    def isovolume(self, min_clim, max_clim, plotter=pv.Plotter(), suppress=False):
        # Plot the isosurface given coordinate arrays X, Y, Z and the field F
        # Use iso_min, iso_max to control the view
        # Create a PyVista structured grid
        grid = pv.StructuredGrid(self.X, self.Y, self.Z)

        # Add the scalar field data to the grid
        grid.point_data["ScalarField"] = self.F.flatten()

        # Define the iso-value for contouring
        # iso_value = 0.1  # Adjust this value based on your data

        # Step 5: Create a plotter object
        # plotter = pv.Plotter()

        # Step 6: Add the volume to the plotter
        # Set opacity to 0.5 for semi-transparency, you can adjust this value
        scalar_bar_args = {
            "title": "None",
            "vertical": True,
            "position_x": 0.85,  # X position of the color bar in the plotter window (0 to 1)
            "position_y": 0.05,  # Y position of the color bar in the plotter window (0 to 1)
            "width": 0.2,  # Width of the color bar
            "height": 0.9,  # Height of the color bar
            "title_font_size": 1,
            "n_labels": 0,  # No numeric labels
            "label_font_size": 1,
            "font_family": "arial",
        }

        # Define hard cutoff
        # cutoff = 0.5
        # scalar_range = np.linspace(0, 1, 256)
        opacity_tf = "linear"

        plotter.add_volume(
            grid,
            scalars="ScalarField",
            opacity=opacity_tf,
            cmap=self.cmap,
            clim=[min_clim, max_clim],
            scalar_bar_args=scalar_bar_args,
        )
        # plotter.add_volume(grid, scalars='ScalarField', cmap = self.cmap, clim = [0, max_clim], scalar_bar_args=scalar_bar_args)

        # Step 7: Extract the surface of the volume
        # surface = grid.threshold(value=0.5, scalars='ScalarField')
        surface = grid.extract_surface()

        # Step 8: Add the surface to the plotter with the same colorscale and lower opacity
        plotter.add_mesh(
            surface,
            scalars="ScalarField",
            opacity=opacity_tf,
            cmap=self.cmap,
            clim=[min_clim, max_clim],
            show_scalar_bar=False,
            show_edges=False,
        )  # Lower opacity for surface
        # plotter.add_mesh(surface, scalars='ScalarField', cmap = self.cmap, clim = [0, max_clim], show_scalar_bar=False, show_edges=False)  # Lower opacity for surface

        contours = grid.contour(np.linspace(min_clim, max_clim, 5)[1:-1])
        plotter.add_mesh(
            contours,
            opacity=0.5,
            cmap=self.cmap,
            clim=[min_clim, max_clim],
            show_scalar_bar=False,
            show_edges=False,
        )

        # Step 9: Set up view
        # plotter.show_axes()
        plotter.view_isometric()
        # plotter.view_xy()
        # plotter.view_yz()
        plotter.set_background("white")

        # Step 10: Show the plot
        if not suppress:
            plotter.show()

    def filled(self, cutoff=0.5, plotter=None, suppress=False):
        # Plot the isosurface given coordinate arrays X, Y, Z and the field F
        # Use cutoff to control the view
        # Create a PyVista structured grid
        grid = pv.StructuredGrid(self.X, self.Y, self.Z)

        # Add the scalar field data to the grid
        grid.point_data["ScalarField"] = self.F.flatten()

        # Define the scalar bar arguments
        scalar_bar_args = {
            "title": "None",
            "vertical": True,
            "position_x": 0.9,  # X position of the color bar in the plotter window (0 to 1)
            "position_y": 0.05,  # Y position of the color bar in the plotter window (0 to 1)
            "width": 0.2,  # Width of the color bar
            "height": 0.9,  # Height of the color bar
            "title_font_size": 1,
            "n_labels": 0,  # No numeric labels
            "label_font_size": 1,
            "font_family": "arial",
        }

        # Define a sharp opacity transfer function for volume rendering
        # Create opacity values that map to scalar field values
        opacity_values = np.linspace(0.0, 1.0, 10)
        opacity_tf = np.where(
            opacity_values < cutoff * np.max(opacity_values), 0.0, len(opacity_values)
        )

        # Add the volume to the plotter
        plotter.add_volume(
            grid,
            scalars="ScalarField",
            opacity=opacity_tf,
            cmap=self.cmap,
            clim=[0.0, 1.0],
            scalar_bar_args=scalar_bar_args,
        )

        # Extract the surface of the volume
        surface = grid.extract_surface()

        # Add the surface to the plotter with scalar opacity (single value)
        plotter.add_mesh(
            surface,
            scalars="ScalarField",
            opacity=opacity_tf,
            cmap=self.cmap,
            clim=[0.0, 1.0],
            show_scalar_bar=False,
            show_edges=False,
        )

        # Add contour lines for better visualization
        contours = grid.contour(cutoff)
        plotter.add_mesh(
            contours,
            opacity=1.0,
            cmap=self.cmap,
            clim=[0.0, 1.0],
            show_scalar_bar=False,
            show_edges=False,
        )

        # Set up the view
        plotter.view_isometric()
        plotter.set_background("white")

        # Show the plot
        if not suppress:
            plotter.show()

    def slice2d(self, plotter=None, normal=(1, 0, 0), origin=None, suppress=False):
        """
        Visualize the scalar field on a structured grid, with optional slicing.

        Parameters:
            cutoff (float): Threshold value for opacity transfer function.
            plotter (pv.Plotter): Optional PyVista plotter instance.
            normal (tuple): Normal vector of the slicing plane.
            origin (tuple): Origin point of the slicing plane. If None, the grid center is used.
        """
        # Create the structured grid
        grid = pv.StructuredGrid(self.X, self.Y, self.Z)
        grid.point_data["ScalarField"] = self.F.flatten()

        # Default origin to center of domain if not provided
        if origin is None:
            origin = (self.X.mean(), self.Y.mean(), self.Z.mean())

        # Define the scalar bar arguments
        scalar_bar_args = {
            "title": "None",
            "vertical": True,
            "position_x": 0.85,  # X position of the color bar in the plotter window (0 to 1)
            "position_y": 0.05,  # Y position of the color bar in the plotter window (0 to 1)
            "width": 0.2,  # Width of the color bar
            "height": 0.9,  # Height of the color bar
            "title_font_size": 1,
            "n_labels": 0,  # No numeric labels
            "label_font_size": 1,
            "font_family": "arial",
        }

        # Slice the grid along the specified plane
        x_max = np.max(self.X)
        y_max = np.max(self.Y)
        z_max = np.max(self.Z)

        # Scale origin from fractions (0-1) to actual coordinates
        if origin is None:
            origin = (x_max / 2, y_max / 2, z_max / 2)
        else:
            origin = (origin[0] * x_max, origin[1] * y_max, origin[2] * z_max)

        slice_plane = grid.slice(normal=normal, origin=origin)
        # slice_plane = grid.slice_orthogonal(x = origin[0], y = origin[1], z = origin[2])
        plotter.add_mesh(
            slice_plane,
            scalars="ScalarField",
            cmap=self.cmap,
            opacity=1,
            clim=[0.0, 1.0],
            show_scalar_bar=True,
            scalar_bar_args=scalar_bar_args,
        )

        # Show coordinate axes
        plotter.show_axes()

        # Face-on view toward the y-z plane
        plotter.view_xy()

        # plotter.view_isometric()
        plotter.set_background("white")

        if not suppress:
            plotter.show()
