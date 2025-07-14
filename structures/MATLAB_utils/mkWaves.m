function [kbz, Gbz, Gsq] = mkWaves(Nx, Ny, Nz, param)
%%% Oliver Xie 2021
%%% Olsen Lab - Massachusetts Institute of Technology
%%%
%%% Subfunction routine to construct all the wavevectors using the
%%% truncation determined by our discretization. All wavevectors are also
%%% shifted into their First Brillouin Zone (FBZ) representation using the
%%% subroutine GtoGbz
%%%
%%% Inputs:
%%%     - Nx, Ny, Nz : Number of points in the unit cell
%%%     - param : struct of parameters
%%%
%%% Outputs:
%%%     - kbz [Nx*Ny*Nz, 3] : wavevector indices corresponding to each
%%%     wavevector. Built by varying first the i index, then j, then k.
%%%     Indices are in the FBZ
%%%     - Gbz [Nx*Ny*Nz, 3] : wavevector values shifted into the FBZ
%%%     - Gsq [Nx*Ny*Nz, 1] : square magnitudes of each wavevector w.r.t
%%%     origin

% Initialization
Gbasis = param.Gbasis;
ijk2h = param.ijk2h;
Nwave = Nx*Ny*Nz;
kbz = zeros(Nwave, 3);
Gbz = zeros(Nwave, 3);
Gsq = zeros(Nwave, 1);
kTemp = [0, 0, 0];     % Temporary measure for each wavevector index

for i = 1:Nx
    for j = 1:Ny
        for k = 1:Nz
            h = ijk2h(i,j,k);
            % Use the FFT convention to determine indices. Indices start at
            % zero. These are unshifted wavevector indices
            kTemp(1) = i - 1;
            kTemp(2) = j - 1;
            kTemp(3) = k - 1;
            % Shift wavevector indices into FBZ using mode 1
            [kbz(h, :), Gsq(h)] = GtoGbz(kTemp, Gbasis, Nx, Ny, Nz, 1, param);
            % Calculate the reciprocal vector corresponding to each
            % indices. The reciprocal basis vector is written as a matrix
            % of [b1; b2; b3] but each in Cartesian coordinates. We write
            % our wavevectors also in Cartesian coordinates, where the x
            % coordinate is the dot product of kbz with all x coordinates
            % of the reciprocal basis (ie the [b1x; b2x; b3x] vector), the
            % y coordinate is the dot product of kbz with all the y
            % coordinates of the reciprocal basis (ie [b1y; b2y; b3y]) and
            % the z coordinate is the dot product of kbz with all the z
            % coordinates of the reciprocal basis (ie [b1z; b2z; b3z])
            for z = 1:3
                Gbz(h, z) = kbz(h, :) * Gbasis(:, z);   % h is wavevector's coordinate in Cartesian
                if abs(Gbz(h, z)) <= 1e-10  % Remove numerical error
                    Gbz(h, z) = 0;
                end
            end
        end
    end
end


end

