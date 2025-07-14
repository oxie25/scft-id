function [ijk2h, h2ijk] = mkIdx(Nx, Ny, Nz)
%%% Oliver Xie 2021
%%% Olsen Lab - Massachusetts Institute of Technology
%%%
%%% Subfunction routine to construct the two mapping indices that go from
%%% [i j k] to [h] and one going from [h] to [i j k]
%%%
%%% Inputs:
%%%     - Nx, Ny, Nz : Number of points in the unit cell
%%%
%%% Outputs:
%%%     - ijk2h [Nx x Ny x Nz] : for every [i j k] point, gives the
%%%     corresponding index in [h]
%%%     - h2ijk [Nx x Ny x Nz, 3] : for every [h] point, gives the
%%%     correspoonding index in [i j k]. Always 3 columns, but if less than
%%%     3 dimensions, then corresponding column in j or k are always 1.

% Anonymous function with the mapping from [i j k] to [h]
hidx = @(i,j,k) (k - 1) * Nx * Ny + (j - 1) * Nx + i;

% Initialize
ijk2h = zeros(Nx, Ny, Nz);
h2ijk = zeros(Nx*Ny*Nz, 3);

% Form the two mapping arrays
for i = 1:Nx
    for j = 1:Ny
        for k = 1:Nz
            h = hidx(i, j, k);
            ijk2h(i, j, k) = h;
            h2ijk(h,:) = [i, j, k];
        end
    end
end

end

