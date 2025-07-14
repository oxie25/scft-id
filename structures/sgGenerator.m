% Copyright Oliver Xie 2025
% Olsen Lab , MIT
% Generates the space group files necessary for running the inverse design
% code by first generating a .mat file containing all structural
% parameters, then calculating .csv and .mat files necessary for various
% functions in the inverse design code.
% 

% Saves:
%   - kbz (first vector of each star) - csv
%   - Gwt (weight of each star) - csv
%   - FS (matrix for transforming between reciprocal to real space) - csv
%   - FT (matrix for transforming between real to reciprocal space) - csv
%   - h2ijk (indexing between the 1D array and the full 3D index, only for plotting) - mat
%   - tauIdx (indexing between the reduced real space group tau and all real space points corresponding to this tau) - mat

% /MATLAB_utils/geoMaker.m which then contains several helper functions
addpath("MATLAB_utils\")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% User defined
symGroup = 'F_d_d_d_1';
numPoints = [32, 32, 32]; % Specify the discretization.
unitCell = 'orthorhombic'; % Specify the unit cell (i.e. cubic, tetragonal, hexagonal, etc...). See MATLAB_utils/groupGeometry.m for a full list of supported unit cells
cellParam = [3, 3, 3]; % The exact numbers here are unimportant, but the dimension should correspond to the number of independent cell parameters for the unit cell (a, b, c, alpha, beta, gamma)
space_group = 'F_d_d_d_1_32_32_32'; % the naming convention should be the space group, followed by the discretization. This is the name of the saved file

% Construct the filepath to the symmetry files
symmetryFolder = fullfile(fileparts(mfilename('fullpath')), 'symmetry');

% Generate a .mat file. This saves a version which must be reloaded
Geometry = geoMaker(numPoints, cellParam, unitCell, symGroup, symmetryFolder);

% Save the .csv and .mat files necessary for the inverse design code. We
% need to package up the inverse design 
Gwt_filename = strcat(space_group, '_Gwt.csv');
kbz_filename = strcat(space_group, '_kbz.csv');
h2ijk_filename = strcat(space_group, '_h2ijk.mat');
tauidx_filename = strcat(space_group, '_tauIdx.mat');
FS_filename = strcat(space_group, '_FS.csv');
FT_filename = strcat(space_group, '_FT.csv');

FS = Geometry.fs; % Only real- if imaginary, something wrong in calc
FT = Geometry.ft; % Only real- if imaginary, something wrong in calc
h2ijk = Geometry.h2ijk;
tauidx = Geometry.tauIdx;
Rwt = cell2mat(Geometry.Rwt)';

if ~Geometry.centro
    % Not centro symmetric
    link = Geometry.link;
    if ~any(link(:))
        % Links are all zero, so closed star
        %%% Number of stars remain the same
        Nstars = Geometry.Nstars;
        kbz = zeros(Nstars, 3);
        Gwt = Geometry.Gwt';
        Gnorm = Geometry.normCoeff;
        for star = 1:Nstars
            % Calculate the perturbation on Gsq when cell parameter is varied
            ktemp = Geometry.kstars{star};
            kbz(star,:) = ktemp(1,:);   % Root wavevector is kbz
        end
    else
        % Links are not all zero, open star
        %%% Need to reduce number of stars
        Nstars = Geometry.Nstars;
        Nstars_mod = Nstars - nnz(link(:,1)) / 2; % Number of links, remove one count of these from Nstars
        Gwt = Geometry.Gwt';

        % The reduced dimension vectors
        kbz_mod = zeros(Nstars_mod, 3);
        Gwt_mod = zeros(Nstars_mod, 1);
        Gnorm_mod = zeros(Nstars_mod, 1);
        % Modify Gwt to remove a single instance of the linked stars
        link_flag = 0; % 0 means first instance of link, 1 means second instance
        real_star_idx = 1;
        for star = 1:Nstars
            if link_flag
                link_flag = 0; % Reset
                continue
            end
            % This is the original stars
            Gwt_mod(real_star_idx) = Gwt(star); % This is real star
            Gnorm_mod(real_star_idx) = Geometry.normCoeff(star);
            ktemp = Geometry.kstars{star}; % This is real star
            kbz_mod(real_star_idx, :) = ktemp(1,:); % Root wavevector is kbz
            real_star_idx = real_star_idx + 1;
            if link(star, 1)
                % If the current link is not 0, add a flag saying next
                % entry is the linked star
                link_flag = 1;
            end
        end
        % Overwrite
        Gwt = Gwt_mod;
        kbz = kbz_mod;
    end
else
    Nstars = Geometry.Nstars;
    kbz = zeros(Nstars, 3);
    Gwt = Geometry.Gwt';
    for star = 1:Nstars
        % Calculate the perturbation on Gsq when cell parameter is varied
        ktemp = Geometry.kstars{star};
        kbz(star,:) = ktemp(1,:);   % Root wavevector is kbz
    end
end
writematrix(Gwt, Gwt_filename)
writematrix(kbz, kbz_filename)
save(h2ijk_filename, "h2ijk")
save(tauidx_filename, "tauidx")
writematrix(FS, FS_filename)
writematrix(FT, FT_filename)