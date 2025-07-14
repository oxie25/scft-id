function Geometry = geoMaker(numPoints, cellParam, unitCell, symGroup, symmetryFolder)
%%% Oliver Xie 2022
%%% Olsen Lab - Massachusetts Institute of Technology
%%%
%%%
%%% Description: Code to set up all the group geometries needed outside the
%%% main code for use in sweeps (reduce calculation overhead per sweep).
%%% Contains whats necessary to generate the star and tau groups.
%%%
%%% Inputs:
%%%     - numPoints : Discretization
%%%     - cellParam : Cell parameters
%%%     - unitCell : Type of crystallographic unit cell
%%%     - symGroup : Space group symmetry
%%%
%%% Outputs:
%%%     A file is saved at the end with the variables necessary in the
%%%     mainSCFT code

Nx = numPoints(1);
Ny = numPoints(2);
Nz = numPoints(3);
param.Nx = Nx;
param.Ny = Ny;
param.Nz = Nz;
if Nz ~= 1 && Ny ~= 1
    dim = 3;
elseif Nz == 1 && Ny ~= 1
    dim = 2;
elseif Nz == 1 && Ny == 1
    dim = 1;
else
    error('Invalid dimension inputted')
end
param.dim = dim;

% Construct two index mapping arrays. One which points to the [i,j,k]
% index given a reduced index [h]. Another which points to the [h] index
% given an array of [i,j,k]
[ijk2h, h2ijk] = mkIdx(Nx, Ny, Nz);
param.ijk2h = ijk2h;
param.h2ijk = h2ijk;

% Generate the symmetries of the unit cell using the subroutine 'rSym'
% Automatically read the symmetries of the unit cell using the subroutine
% symExtract
%[Rsym, Tsym, centro] = rSym(symGroup);
[Rsym, Tsym, centro, shifted_origin, inversion_translation] = symExtract(symGroup, dim, symmetryFolder);
symnum = length(Rsym); % Number of symmetry operators
param.Rsym = Rsym;
param.Tsym = Tsym;
param.centro = centro;
param.sym = symnum;
param.shifted_origin = shifted_origin;
param.inversion_translation = inversion_translation;

% Read cell parameters
numCP = size(cellParam,1);
param.numCP = numCP;

% Calculate the Bravais basis vectors and derivatives
[Rbasis, dRbasis] = groupGeometry(unitCell, cellParam);

% Calculate the reciprocal basis vectors
b1 = cross(Rbasis(2,:), Rbasis(3,:)) ./ dot( Rbasis(1,:), cross( Rbasis(2,:), Rbasis(3,:) ) );
b2 = cross(Rbasis(3,:), Rbasis(1,:)) ./ dot( Rbasis(1,:), cross( Rbasis(2,:), Rbasis(3,:) ) );
b3 = cross(Rbasis(1,:), Rbasis(2,:)) ./ dot( Rbasis(1,:), cross( Rbasis(2,:), Rbasis(3,:) ) );
Gbasis = (2*pi) * [b1; b2; b3];
% Calculate the volume
V = dot( Rbasis(1,:), cross( Rbasis(2,:), Rbasis(3,:) ) );
% Store values in param
param.Rbasis = Rbasis;
param.dRbasis = dRbasis;
param.Gbasis = Gbasis;
param.V = V;

% Generate all FBZ-shifted wavevectors and wavevector indices using 'mkWaves'
% Also generate the square magnitude of each wavevector
[kbz, Gbz, Gsq] = mkWaves(Nx, Ny, Nz, param);
Nwave = size(Gbz, 1);   % Number of waves

% Sort all the wavevectors in order by their square magnitudes
[GsqSort, GidxSort] = sort(Gsq);
kbzSort = kbz(GidxSort, :);
GbzSort = Gbz(GidxSort, :);
% Assign to param struct
param.Nwave = Nwave;
param.Gbz = Gbz;
param.kbz = kbz;
param.Gsq = Gsq;
param.GidxSort = GidxSort;    % This remains invariant throughout all iterations

% Generate all stars and star coefficients using 'mkStars'
[Gwt, kstars, Gstars, Nstars, starIdx, Tstars, normCoeff, link, symopstar, TinvPhase] = mkStars(GbzSort, GsqSort, kbzSort, param);
% Find the phase factors
starPhase = phaseFind(Gbasis, Rbasis, normCoeff, Tstars, kstars, Nstars);
% Find the squared magnitude of each star (for MDE solving)
% Note: if there are linked stars, we should only report a single star
% Satisfies centrosymmetry - always calculate
GsqStar = zeros(Nstars, 1);
for i = 1:Nstars
    GsqStar(i) = Gsq( GidxSort( starIdx{i}(1) ) );
end

if ~centro
    % No longer satisfies centrosymmetry
    if ~any(link(:))
        % Links are all zero, so closed star
        % GsqStar calculation proceeds normally
        GsqStar = zeros(Nstars, 1);
        for i = 1:Nstars
            GsqStar(i) = Gsq( GidxSort( starIdx{i}(1) ) );
        end
    else
        % Links are not all zero, open star
        % Only calculate GsqStar for unlinked stars, and once for a linked
        % star
        Nstars_mod = Nstars - nnz(link(:,1)) / 2; % Number of links, remove one count of these from Nstars
        GsqStar_nocentro = zeros(Nstars_mod, 1);
        star_idx = 1;
        link_flag = 0;
        for i = 1:Nstars
            if link_flag
                % We just found a linked star, skip this star
                link_flag = 0; % Reset flag
                continue
            end
            
            if link(i, 1)
                % Link is not 0, there is another attached star
                GsqStar_nocentro(star_idx) = Gsq( GidxSort( starIdx{i}(1) ) );
                star_idx = star_idx + 1;
                link_flag = 1;
            else
                % Link is 0, calculate normally
                GsqStar_nocentro(star_idx) = Gsq( GidxSort( starIdx{i}(1) ) );
                star_idx = star_idx + 1;
            end
        end
        param.GsqStar_nocentro = GsqStar_nocentro;
    end
end
% Assign to param struct
param.Gwt = Gwt;
param.kstars = kstars;
param.Gstars = Gstars;
param.starIdx = starIdx;
param.normCoeff = normCoeff;
param.Nstars = Nstars;
param.Tstars = Tstars;
param.link = link;
param.phase = starPhase;
param.GsqStar = GsqStar;
param.symopstar = symopstar;


%%% New mkTaus (handles glide planes)
%%% Ridx_store and Ridx_flag not needed but is good check to make sure all
%%% points have been assigned
[ptaus, tauIdx, symoptau, Rwt, Ridx_store, Ridx_flag, R] = mkTaus(numPoints, param);
if any(~Ridx_flag)
    warning('Not all points assigned to tau group')
end
Ntaus = length(ptaus);
% Assign to param struct
param.Ntaus = Ntaus;
param.ptaus = ptaus;
param.tauIdx = tauIdx;
% param.RsqTau = RsqTau;
param.symoptau = symoptau;
param.Rwt = Rwt;
% param.Ridxsort = Ridxsort;
param.R = R;
param.TinvPhase = TinvPhase;


%%% Precalculate necessary values for the crystallographic Fourier
%%% Transform
[ft, fs, s_idx] = fourierCalc_v2(kstars, ptaus, numPoints, param);
param.ft = ft;
param.fs = fs;
idx = s_idx;
param.idx = idx;

Geometry = param;

if any(imag(ft), 'all') || any(imag(fs), 'all')
    fprintf('Imaginary detected with space group %s \n', symGroup)
end

if ~conversion_test(fs, ft)
    fprintf('Possible bad fs and ft calculated with space group %s \n', symGroup)
end

% Do a quick conversion test: generate a random field, convert from real to
% reciprocal back to real, then again, to see if it works

filename = strcat(symGroup, '_', num2str(Nx), '_', num2str(Ny), '_', num2str(Nz), '.mat');
save(filename, 'Geometry', '-v7.3');

end

function result = conversion_test(fs, ft)
Ntau = size(fs, 1);
Nstar = size(ft, 1);
real_field_init = rand(Ntau, 1);
star_field = ft * real_field_init;
real_field = fs * star_field;
% Convert again in case we have unallowed non-symmetries in the real field
% init

star_field = ft * real_field;
real_field_end = fs * star_field;

if norm(real_field_end - real_field) / norm(real_field_end) < 1e-12
    result = 1;
else
    result = 0;
end

end

