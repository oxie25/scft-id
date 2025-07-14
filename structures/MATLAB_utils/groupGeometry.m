function [Rbasis, dRbasis] = groupGeometry(unitCell, cellParam)
%%% Oliver Xie 2021
%%% Olsen Lab - Massachusetts Institute of Technology
%%%
%%% Subfunction routine to construct the unit cell based on the geometry.
%%% Reads in the variable 'unitCell' and returns matrix denoting the
%%% Bravais basis vectors in direct space constructed as [a1; a2; a3] in
%%% Cartesian space where a1 = [x1, y1, z1] and so forth. Also returns the
%%% cell array dRbasis which returns all matrices representing the
%%% derivative of the normalized basis matrix with respect to all
%%% independent cell parameters governed by the geometry. Matrices are
%%% returned according to the dimensionality of the geometry. Un-needed
%%% dimensions are returned as a vector of zeros in the matrix.
%%% We encode all the Bravais lattices disregarding duplications due to
%%% centering operations

%%% Inputs:
%%%     - unitCell : tells function the geometry of the problem
%%%     - cellParam : gives the cell parameters to be varied. Must be
%%%     constructed as a vector with lengths first and angles following.
%%%     See each case for the exact configuration this variable must be in.
%%%
%%% Outputs:
%%%     - Rbasis [3 x 3] : The basis vectors of the cell given in Cartesian
%%%     coordinates. Each row corresponds to a different basis in the
%%%     convention of [a1; a2; a3] while each column is Cartesian
%%%     coordinate [x, y, z] respectively
%%%     - dRbasis (cell array of [3 x 3]) : The derivatives of the basis
%%%     vectors with respect to every independent cell parameter. Given as
%%%     a cell array if more than one unit cell parameter. Each derivative
%%%     matrix is in the same convention as Rbasis in Cartesian
%%%     coordinates

switch unitCell
    %%% 1D Cases
    case 'lam'          % Lamellar
        L1 = cellParam(1);
        Rbasis = [L1, 0, 0; 0, 1, 0; 0, 0, 1];
        dRbasis{1} = [1, 0, 0; 0, 0, 0; 0, 0, 0];    % w.r.t L1

    %%% 2D Cases
    case 'oblique'
        L1 = cellParam(1);
        L2 = cellParam(2);
        gamma = cellParam(3);
        Rbasis = [L1, 0, 0; L2 * cos(gamma), L2 * sin(gamma), 0; 0, 0, 1];
        dRbasis{1} = [1, 0, 0; 0, 0, 0; 0, 0, 0];
        dRbasis{2} = [0, 0, 0; cos(gamma), sin(gamma), 0; 0, 0, 0];
        dRbasis{3} = [0, 0, 0; -L2 * sin(gamma), L2 * cos(gamma), 0; 0, 0, 0];
    case 'rectangular'
        L1 = cellParam(1);
        L2 = cellParam(2);
        Rbasis = [L1, 0, 0; 0, L2, 0; 0, 0, 1];
        dRbasis{1} = [1, 0, 0; 0, 0, 0; 0, 0, 0];
        dRbasis{2} = [0, 0, 0; 0, 1, 0; 0, 0, 0];
    case 'square'       % Square
        L1 = cellParam(1);
        Rbasis = [L1, 0, 0; 0, L1, 0; 0, 0, 1];
        dRbasis{1} = [1, 0, 0; 0, 1, 0; 0, 0, 0];   % w.r.t L1
    case 'hex'          % Hexagonal
        L1 = cellParam(1);
        Rbasis = [L1, 0, 0; L1 * cos(2*pi/3), L1 * sin(2*pi/3), 0; 0, 0, 1];
        dRbasis{1} = [1, 0, 0; cos(2*pi/3), sin(2*pi/3), 0; 0, 0, 0];     % w.r.t L1

    %%% 3D Cases
    case 'triclinic'
        L1 = cellParam(1);
        L2 = cellParam(2);
        L3 = cellParam(3);
        alpha = cellParam(4);
        beta = cellParam(5);
        gamma = cellParam(6);
        cos_alpha = cos(alpha);
        sin_alpha = sin(alpha);
        cos_beta = cos(beta);
        sin_beta = sin(beta);
        cos_gamma = cos(gamma);
        sin_gamma = sin(gamma);

        f = 1 - cos_alpha^2 - cos_beta^2 - cos_gamma^2 + 2 * cos_alpha * cos_beta * cos_gamma;
        V = L1 * L2 * L3 * sqrt(f);
        dV_alpha = L1 * L2 * L3 / sqrt(f) * sin_alpha * (cos_alpha - cos_beta * cos_gamma);
        dV_beta = L1 * L2 * L3 / sqrt(f) * sin_beta * (cos_beta - cos_alpha * cos_gamma);
        dV_gamma = L1 * L2 * L3 / sqrt(f) * sin_gamma * (cos_gamma - cos_alpha * cos_beta);


        Rbasis = [L1, 0, 0; 
            L2 * cos_gamma, L2 * sin_gamma, 0; 
            L3 * cos_beta, L3 * (cos_alpha - cos_beta * cos_gamma) / sin_gamma, V / (L1 * L2 * sin_gamma)];
        dRbasis{1} = [1, 0, 0; 0, 0, 0; 0, 0, 0];
        dRbasis{2} = [0, 0, 0; cos_gamma, sin_gamma, 0; 0, 0, 0];
        dRbasis{3} = [0, 0, 0; 0, 0, 0; cos_beta, (cos_alpha - cos_beta * cos_gamma) / sin_gamma, V / (L1 * L2 * L3 * sin_gamma)];
        dRbasis{4} = [0, 0, 0; 0, 0, 0; 0, -L3 * sin_alpha, dV_alpha / (L1 * L2 * sin_gamma)];
        dRbasis{5} = [0, 0, 0; 0, 0, 0; -L3 * sin_beta, L3 * sin_beta * cos_gamma, dV_beta / (L1 * L2 * sin_gamma)];
        dRbasis{6} = [0, 0, 0; -L2 * sin_gamma, L2 * cos_gamma, 0; 0, L3 * cos_beta * sin_gamma, dV_gamma / (L1 * L2 * sin_gamma) - V / (L1 * L2 * sin_gamma^2) * cos_gamma];

    case 'monoclinic'
        L1 = cellParam(1);
        L2 = cellParam(2);
        L3 = cellParam(3);
        alpha = pi/2;
        beta = cellParam(4);
        gamma = pi/2;
        cos_alpha = cos(alpha);
        cos_beta = cos(beta);
        sin_beta = sin(beta);
        cos_gamma = cos(gamma);
        sin_gamma = sin(gamma);

        f = 1 - cos_alpha^2 - cos_beta^2 - cos_gamma^2 + 2 * cos_alpha * cos_beta * cos_gamma;
        V = L1 * L2 * L3 * sqrt(f);
        dV_beta = L1 * L2 * L3 / sqrt(f) * sin_beta * (cos_beta - cos_alpha * cos_gamma);

        Rbasis = [L1, 0, 0; 
            L2 * cos_gamma, L2 * sin_gamma, 0; 
            L3 * cos_beta, L3 * (cos_alpha - cos_beta * cos_gamma) / sin_gamma, V / (L1 * L2 * sin_gamma)];
        dRbasis{1} = [1, 0, 0; 0, 0, 0; 0, 0, 0];
        dRbasis{2} = [0, 0, 0; cos_gamma, sin_gamma, 0; 0, 0, 0];
        dRbasis{3} = [0, 0, 0; 0, 0, 0; cos_beta, (cos_alpha - cos_beta * cos_gamma) / sin_gamma, V / (L1 * L2 * L3 * sin_gamma)];
        dRbasis{4} = [0, 0, 0; 0, 0, 0; -L3 * sin_beta, L3 * sin_beta * cos_gamma, dV_beta / (L1 * L2 * sin_gamma)];
        
    case 'rhombohedral' % Rhombohedral
        L1 = cellParam(1);
        L2 = L1;
        L3 = L1;
        alpha = cellParam(2);
        beta = alpha;
        gamma = alpha;
        cos_alpha = cos(alpha);
        sin_alpha = sin(alpha);
        cos_beta = cos(beta);
        sin_beta = sin(beta);
        cos_gamma = cos(gamma);
        sin_gamma = sin(gamma);

        f = 1 - cos_alpha^2 - cos_beta^2 - cos_gamma^2 + 2 * cos_alpha * cos_beta * cos_gamma;
        V = L1 * L2 * L3 * sqrt(f);
        dV_alpha = -L1 * L2 * L3 / sqrt(f) * 3 * sin_alpha * (cos_alpha - 1) * cos_alpha;

        Rbasis = [L1, 0, 0; 
            L2 * cos_gamma, L2 * sin_gamma, 0; 
            L3 * cos_beta, L3 * (cos_alpha - cos_beta * cos_gamma) / sin_gamma, V / (L1 * L2 * sin_gamma)];
        dRbasis{1} = [1, 0, 0; cos_gamma, sin_gamma, 0; cos_beta, (cos_alpha - cos_beta * cos_gamma) / sin_gamma, V / (L1 * L2 * L3 * sin_gamma)];
        dRbasis{2} = [0, 0, 0; -L2 * sin_gamma, L2 * cos_gamma, 0; -L3 * sin_beta, L3 * sin_alpha * (2 * cos_alpha - 1), dV_alpha / (L1 * L2 * sin_gamma) - V / (L1 * L2 * L3 * sin_gamma^2) * cos_alpha];

    case 'hexagonal' % Hexagonal
        L1 = cellParam(1);
        L2 = L1;
        L3 = cellParam(2);
        alpha = pi/2;
        beta = pi/2;
        gamma = 2*pi/3;
        cos_alpha = cos(alpha);
        cos_beta = cos(beta);
        cos_gamma = cos(gamma);
        sin_gamma = sin(gamma);

        f = 1 - cos_alpha^2 - cos_beta^2 - cos_gamma^2 + 2 * cos_alpha * cos_beta * cos_gamma;
        V = L1 * L2 * L3 * sqrt(f);

        Rbasis = [L1, 0, 0; 
            L2 * cos_gamma, L2 * sin_gamma, 0; 
            L3 * cos_beta, L3 * (cos_alpha - cos_beta * cos_gamma) / sin_gamma, V / (L1 * L2 * sin_gamma)];
        dRbasis{1} = [1, 0, 0; cos_gamma, sin_gamma, 0; 0, 0, 0];
        dRbasis{2} = [0, 0, 0; 0, 0, 0; cos_beta, (cos_alpha - cos_beta * cos_gamma) / sin_gamma, V / (L1 * L2 * L3 * sin_gamma)];
        
    case 'orthorhombic' % Orthorhombic
        L1 = cellParam(1);
        L2 = cellParam(2);
        L3 = cellParam(3);
        Rbasis = [L1, 0, 0; 0, L2, 0; 0, 0, L3];
        dRbasis{1} = [1, 0, 0; 0, 0, 0; 0, 0, 0];
        dRbasis{2} = [0, 0, 0; 0, 1, 0; 0, 0, 0];
        dRbasis{3} = [0, 0, 0; 0, 0, 0; 0, 0, 1];

    case 'tetragonal' % tetragonal
        L1 = cellParam(1);
        L2 = cellParam(2);
        Rbasis = [L1, 0, 0; 0, L1, 0; 0, 0, L2];
        dRbasis{1} = [1, 0, 0; 0, 1, 0; 0, 0, 0];
        dRbasis{2} = [0, 0, 0; 0, 0, 0; 0, 0, 1];
        
    case 'cubic'        % Cubic
        L1 = cellParam(1);
        Rbasis = [L1, 0, 0; 0, L1, 0; 0, 0, L1];
        dRbasis{1} = [1, 0, 0; 0, 1, 0; 0, 0, 1];      % w.r.t L1
        
    otherwise
        error('Cell shape not found')
end

% Remove numerical errors
tolerance = 1e-10;
Rbasis(abs(Rbasis) < tolerance) = 0;
for i = length(dRbasis)
    dRactive = dRbasis{i};
    dRactive(abs(dRactive) < tolerance) = 0;
    dRbasis{i} = dRactive;
end

end

