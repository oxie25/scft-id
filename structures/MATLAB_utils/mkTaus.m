function [ptau, tauIdx, symop, wt, idx_store, idx_flag, R] = mkTaus(numPoints, param)
%%% Oliver Xie 2022
%%% Olsen Lab - Massachusetts Institute of Technology
%%%
%%% Subfunction routine to generate the set of taus that contain all
%%% equivalent real space points
%%%
%%% Inputs:
%%%     - numPoints [4, 1] : Array containing the discretization of the
%%%     system where the first three values are spatial x y z
%%%     - param : Struct of parameters
%%%
%%% Outputs:
%%%     - ptau {Ntau, 1} : Cell array containing the grouping of vector
%%%     coefficients [p1 p2 p3] for vectors that exist in a tau
%%%     - tauIdx {Ntau, 1} : Cell array containing the index of waves in
%%%     the sorted list format of each wave in a tau
%%%     - symop {Ntau, 1} : Cell array containing the symmetry operation
%%%     that brings each root vector in a tau to all other points in the
%%%     tau
%%%     - wt [Ntau, 1] : Array containing the number of waves in a tau
%%%     - idx_store [Nwaves, 1] : Array storing the internal indexing of
%%%     how each R point appears
%%%     - idx_flag [Nwaves, 1] : Array storing the flag of each index
%%%     showing if it has been assigned to a tau group yet
%%%     - R [Nwaves, 3] : Array containing the [p1 p2 p3] coefficients for
%%%     the real space waves

%%% Note: all indices are in crystallographic coordinates
%%% Construct the vectors describing every point in real space. Only in
%%% crystallographic coordinates
%%% idx_flag is vector of checked indices
%%% idx_store is vector of the index of the wavevector
M1 = numPoints(1);
M2 = numPoints(2);
M3 = numPoints(3);
R = zeros(M1 * M2 * M3, 3);
Rsq = zeros(M1 * M2 * M3, 1);
idx_flag = zeros(M1 * M2 * M3, 1);
idx_store = zeros(M1 * M2 * M3, 1);
idx = 1;

Rbasis = param.Rbasis;

for p3 = 1:M3
    for p2 = 1:M2
        for p1 = 1:M1
            % Remember index begins at 0
            if (p1 - 1) > M1 / 2
                p1v = p1 - M1;
            else
                p1v = p1;
            end
            if (p2 - 1) > M2 / 2
                p2v = p2 - M2;
            else
                p2v = p2;
            end
            if (p3 - 1) > M3 / 2
                p3v = p3 - M3;
            else
                p3v = p3;
            end
            R(idx, :) = [p1v-1, p2v-1, p3v-1];
            for i = 1:3
                v(i) = R(idx, :) * Rbasis(:,i);
            end
            Rsq(idx, :) = norm(v);
            idx_store(idx) = idx;
            idx = idx + 1;
        end
    end
end

%%% Loop over every crystallographic vector, check if it has already been
%%% assigned a tau group, create its tau group if it has not
Rsym = param.Rsym;
Tsym = param.Tsym;
tau_idx = 1;
for v_idx = 1:size(R, 1)
    if idx_flag(v_idx) == 1
        % Skip
        continue
    else
        % Create tau group
        % Turn the current idx_flag to 1
        idx_flag(v_idx) = 1;
        [ptau_temp, tauIdx_temp, symop_temp, wt_temp, idx_flag] = sym_finder(param, R(v_idx, :), idx_store, idx_flag, Rsym, Tsym, v_idx);
        % Assign to cell arrays
        ptau{tau_idx} = ptau_temp;
        tauIdx{tau_idx} = tauIdx_temp;
        symop{tau_idx} = symop_temp;
        wt{tau_idx} = wt_temp;
        % Increase tau index
        tau_idx = tau_idx + 1;
    end
end
end

function [ptau, tauIdx, symop, wt, idx_flag] = sym_finder(param, R_cur, idx_store, idx_flag, Rsym, Tsym, v_idx)
% Subfunction that generates all the taus of a given vector R_cur
% Identifies a vector is symmetry accessible IF the resulting
% crystallographic coordinate are all integers
N_sym = length(Rsym);
Nx = param.Nx;
Ny = param.Ny;
Nz = param.Nz;
% Store the current wave as 1 of the tau
ptau(1,:) = R_cur;
tauIdx(1,:) = v_idx;
symop(1,:) = 1; % Identity operator

% Start tau_idx at 2
tau_idx = 2;
flag = 0;

for sym_idx = 2:N_sym
    % Always skip the identity symmetry operator
    R_test = Rsym{sym_idx} * R_cur' + (Tsym{sym_idx} .* [Nx, Ny, Nz])';
    %%% Condition slightly to prevent rounding error, also if all three
    %%% cases are true, then this is a good crystrallographic vector
    if norm(round(R_test(1)) - R_test(1)) < 1E-9
        R_test(1) = round(R_test(1));
        flag = flag + 1;
    end
    
    if norm(round(R_test(2)) - R_test(2)) < 1E-9
        R_test(2) = round(R_test(2));
        flag = flag + 1;
    end
    
    if norm(round(R_test(3)) - R_test(3)) < 1E-9
        R_test(3) = round(R_test(3));
        flag = flag + 1;
    end
    if flag == 3
        % The tested vector is an integer, test to see if already found
        R_test_bz = RtoRbz(R_test, Nx, Ny, Nz);
        % Uncovert from the [-N to N] index to the [0 to N] index to
        % quickly flag the correct crystallographic wavevector that is
        % found
        R_full = RbztoR(R_test_bz, Nx, Ny, Nz);
        % Generates the index value, should correspond exactly with how the
        % vectors were generated
        R_value = (R_full(1) + 1) + (R_full(2)) * Nx + (R_full(3)) * Nx * Ny;
        % Check to see if this point has been arrived at already (flag)
        if idx_flag(R_value) == 1
            % already found
            flag = 0; % Reset flag!
            continue
        else
            idx_flag(R_value) = 1;
            % Store found vector as part of tau
            ptau(tau_idx,:) = R_test_bz;
            % Store tauIdx, 
            tauIdx(tau_idx, :) = idx_store(R_value);
            symop(tau_idx, :) = sym_idx;
            % Increment tau
            tau_idx = tau_idx + 1;
            % Reset the flag 
            flag = 0;
        end
    else
        % Reset the flag
        flag = 0;
    end 
end
wt = tau_idx - 1;
end

function ptrans = RbztoR(pbztrans, Nx, Ny, Nz)
%%% Shifts a vector in the 1st brillouin zone into a vector with [0 to N]
%%% indexing
p1 = pbztrans(1);
p2 = pbztrans(2);
p3 = pbztrans(3);

if p1 < 0
    p1v = p1 + Nx;
else
    p1v = p1;
end

if p2 < 0
    p2v = p2 + Ny;
else
    p2v = p2;
end

if p3 < 0
    p3v = p3 + Nz;
else
    p3v = p3;
end

ptrans = [p1v, p2v, p3v];

end

function pbztrans = RtoRbz(ptrans, Nx, Ny, Nz)
%%% Shifts a vector of [p1 p2 p3] into unit cell constrained in index -M/2
%%% + 1 to M/2

p1 = ptrans(1);
p2 = ptrans(2);
p3 = ptrans(3);

%%% Condition slightly to prevent rounding error
if norm(round(p1) - p1) < 1E-9
    p1 = round(p1);
end

if norm(round(p2) - p2) < 1E-9
    p2 = round(p2);
end

if norm(round(p3) - p3) < 1E-9
    p3 = round(p3);
end

p1shift = p1;
p2shift = p2;
p3shift = p3;

%%% Check x index
if p1 < -(Nx / 2 + 1)           % less than box
    p1shift = p1 + Nx;
elseif p1 > Nx / 2     % more than box
    p1shift = p1 - Nx;
end
%%% Check y index
if p2 < -(Ny / 2 + 1)           % less than box
    p2shift = p2 + Ny;
elseif p2 > Ny / 2     % more than box
    p2shift = p2 - Ny;
end
%%% Check z index
if p3 < -(Nz / 2 + 1)           % less than box
    p3shift = p3 + Nz;
elseif p3 > Nz / 2     % more than box
    p3shift = p3 - Nz;
end

pbztrans = [p1shift, p2shift, p3shift];

end
