function Gbz = GbzAll(G, Gbasis, Nx, Ny, Nz)
%%% Oliver Xie 2021
%%% Olsen Lab - Massachusetts Institute of Technology
%%%
%%% Subfunction routine to find ALL equivalent representations of a
%%% wavevector one Nx, Ny, and Nz away in all directions
%%%
%%% Inputs:
%%%     - G : Current wavevector
%%%     - Gbasis : Reciprocal basis vector
%%%     - Nx : A period in the i index
%%%     - Ny : A period in the j index
%%%     - Nz : A period in the k index
%%%
%%% Outputs:
%%%     - Gbz : Matrix of all the wavevector indices [i, j, k] that are
%%%     accesible as 1 period away in any direction from the input wavevector 

Gsq_min = 1E10;
delta = 1E-8;
k = 1;

for i1 = 1:-1:-1
    G_try(1) = G(1) + i1*Nx;
    for i2 = 1:-1:-1
        G_try(2) = G(2) + i2*Ny;
        for i3 = 1:-1:-1
            G_try(3) = G(3) + i3*Nz;
            for i = 1:3
                v(i) = G_try(:)' * Gbasis(:,i);
            end
            Gsq = v * v(:);
            if Gsq < Gsq_min
                G_min(k,:) = G_try;
                Gsq_min = Gsq;
                k = k + 1;
            elseif Gsq == Gsq_min   % Assign all equivalent representations on boundaries
                G_min(k,:) = G_try;
                k = k + 1;
            end
        end
    end
end
Gbz = G_min;

end