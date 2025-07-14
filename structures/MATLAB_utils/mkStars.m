function [wt, kstars, Gstars, Nstars, starIdx, Tstars, coeff, link, symop, TinvPhase] = mkStars(GbzSort, GsqSort, kbzSort, param)
%%% Oliver Xie 2021
%%% Olsen Lab - Massachusetts Institute of Technology
%%%
%%% Subfunction routine to generate the set of stars that represent all
%%% unique wavevectors. All other wavevectors can be built from their
%%% respective stars using the symmetries of the space group. This is
%%% equivalent to the Irreducible Brillouin Zone representation (IBZ).
%%% Symmetry operations are checked in CRYSTALLOGRAPHIC COORDINATES
%%% (03-08-2022 change)
%%% Symmetry operations were not being checked in crystallographic
%%% coordinates, changed because GbzAll and GtoGbz need the wavevector
%%% indices to work
%%% (12-19-2024 change)
%%%
%%% Inputs:
%%%     - GbzSort [Nx*Ny*Nz, 3] : Array containing every wavevector in the
%%%     FBZ and sorted according to the square magnitude Gsq (small to
%%%     large)
%%%     - GsqSort [Nx*Ny*Nz, 1] : Vector containing the square magnitudes
%%%     of all the wavevectors sorted from smallest to largest
%%%     - kbzSort [Nx*Ny*Nz, 3] : Array containing every wavevector index
%%%     in the FBZ and sorted according to the square magnitude Gsq (small
%%%     to large)
%%%
%%% Outputs:
%%%     - wt [Nstars, 1] : Number of wavevectors that can be built from the
%%%     root wavevector of each star
%%%     - kstars {Nstars} : Cell array containing the wavevector indices of
%%%     every wavevector contained within a star
%%%     - Gstars {Nstars} : Cell array containing the wavevector values of
%%%     every wavevector contained within a star
%%%     - Nstars : Number of stars found
%%%     - starIdx {Nstars} : Cell array containing the index of each
%%%     wavevector in the sorted list
%%%     - Tstars {Nstars} : Cell array containing the translational
%%%     component of the symmetry operator that brings each root within a
%%%     star to the other accessible wavevectors in the star
%%%     - coeff [Nstars, 1] : Coefficients of each star. Coefficients of
%%%     cancelled stars are zero
%%%     - link [Nstars, 2] : Array containing non-zero values if there is a
%%%     link between two stars to form a conjugate basis (see the open and
%%%     non-centrosymmetric case)
%%%     - symop {Nstars} : Cell array containing the symmetry operation
%%%     used in the star
%%%     - TinvPhase {Nstars} : The phase relationship if we have
%%%     non-centrosymmetric but closed space groups

% Load Parameters
centro = param.centro;
shifted_origin = param.shifted_origin;
Nx = param.Nx;
Ny = param.Ny;
Nz = param.Nz;
Gbasis = param.Gbasis;
Rbasis = param.Rbasis;
Nwave = param.Nwave;
epsilon = 1e-10;

% Origin (k = [0 0 0]) is always in its own star
wt(1,:) = 1;
kstars{1} = kbzSort(1,:);
Gstars{1} = GbzSort(1,:);
Tstars{1} = [0, 0, 0];
symop{1} = 1;
Gsq_stars = GsqSort(1);
Nstars = 1;
starIdx{1} = 1;   % Tracking the index of waves of the sorted list in each star
first = 1;

% Identify places where Gsq_sort changes values
Gsq_temp = 0;
for i = 2:Nwave
    if GsqSort(i) > Gsq_temp + epsilon   % Current ||Gbz|| is greater than previous
        % Check if all waves in star can be reached by symmetry of the
        % root. If not all waves are reached by group symmetry, call
        % deconvolute_star until found
        if ~first   % Check that this is not the first non-zero wave being allocated
            % Check if the wave being assigned to a star is actually formed by
            % symmetry operations or just has the same magnitude. If is not
            % actually part of star, call subfunction 'deconvolute_star'
            % at the end of this set of magnitudes to identify all separate
            % stars. Use subfunction 'is_symmetrical'; returns a vector
            % comparing all waves in kstars_temp against the first (root)
            % wave in kstars_temp
            [symmetry, Tstars_temp, symop_temp] = is_symmetrical(kstars_temp, param);
            if ~all(symmetry)
                [Gstars_temp, kstars_temp, star_idx_temp, Tstars_temp, symop_temp] = deconvolute_star(Gstars_temp, kstars_temp, star_idx_temp, param);
                % Allocate the deconvoluted waves into their own stars
                % (Gstars_temp is now a cell array)
                add_stars = size(Gstars_temp, 2);   % Additional stars found
                for j = 1:add_stars % Allocate each star into permanent measure
                    wt(Nstars + j - 1) = size(Gstars_temp{j}, 1);
                    kstars{Nstars + j - 1} = kstars_temp{j};
                    Gstars{Nstars + j - 1} = Gstars_temp{j};
                    starIdx{Nstars + j - 1} = star_idx_temp{j};
                    Tstars{Nstars + j - 1} = Tstars_temp{j};
                    Gsq_stars(Nstars + j - 1) = GsqSort(i);
                    symop{Nstars + j - 1} = symop_temp{j};
                end
                Nstars = Nstars + add_stars - 1;
            else % If symmetry obeyed in a star, allocate to permanent measure
                wt(Nstars) = wt_temp;
                kstars{Nstars} = kstars_temp;
                Gstars{Nstars} = Gstars_temp;
                starIdx{Nstars} = star_idx_temp;
                Tstars{Nstars} = Tstars_temp;
                Gsq_stars(Nstars) = GsqSort(i);
                symop{Nstars} = symop_temp;
            end
        end
        % Reset all temporary measures (only valid within a star)
        clear wt_temp Gstars_temp kstars_temp star_idx_temp symmetry Tstars_temp
        Gsq_temp = GsqSort(i); % Update the comparison value
        Nstars = Nstars + 1;  % Update the number of stars found
        wt_temp = 1;            % Set weight as 1 (first wave in star)
        star_idx_temp(wt_temp,:) = i;            % Store current index value as first index in star
        kstars_temp(wt_temp,:) = kbzSort(i,:);  % Store current kbz value as first wave in star
        Gstars_temp(wt_temp,:) = GbzSort(i,:);  % Store current Gbz value as first wave in star
        if i == Nwave     % Last wave reached AND also new ||Gbz|| detected - ensure we aren't double assigning
            % If this is also the last wavevector, then we can directly
            % assign it to its own star without any symmetry checks
            % Allocate all into the permanent measure directly (no need to
            % check symmetry because no other waves)
            wt(Nstars) = 1;
            kstars{Nstars} = kbzSort(i,:);
            Gstars{Nstars} = GbzSort(i,:);
            starIdx{Nstars} = i;
            Tstars{Nstars} = [0, 0, 0];    % If only one wave present in a star, then only the identity operator is a symmetry operator
            Gsq_stars(Nstars) = GsqSort(i);
            symop{Nstars} = 1;             % Only identity operator present
        end
        
    elseif i ~= Nwave
        wt_temp = wt_temp + 1;  % Increase number of waves found
        star_idx_temp(wt_temp,:) = i;            % Store current index value as an index in star
        kstars_temp(wt_temp,:) = kbzSort(i,:);  % Store current kbz value as a wave in star
        Gstars_temp(wt_temp,:) = GbzSort(i,:);  % Store current Gbz value as a wave in star
        first = 0;  % No longer first non-zero wave being checked
        
    elseif i == Nwave     % Last wave reached and not new Gsq value
        % Check this
        wt_temp = wt_temp + 1;
        star_idx_temp(wt_temp,:) = i;            % Store current index value as an index in star
        kstars_temp(wt_temp,:) = kbzSort(i,:);  % Store current kbz value as a wave in star
        Gstars_temp(wt_temp,:) = GbzSort(i,:);  % Store current Gbz value as a wave in star
        % Check if the wave being assigned to a star is actually formed by
        % symmetry operations or just has the same magnitude. If is not
        % actually part of star, call subfunction 'deconvolute_star'
        % at the end of this set of magnitudes to identify all separate
        % stars. Use subfunction 'is_symmetrical'; returns a vector
        % comparing all waves in Gstars_temp against the first (root)
        % wave in Gstars_temp as well as the translation operation
        % associated with this symmetry operation (Tstars_temp)
        [symmetry, Tstars_temp, symop_temp] = is_symmetrical(kstars_temp, param);
        if ~all(symmetry)
            [Gstars_temp, kstars_temp, star_idx_temp, Tstars_temp, symop_temp] = deconvolute_star(Gstars_temp, kstars_temp, star_idx_temp, param);
            % Allocate the deconvoluted waves into their own stars
            % (Gstars_temp is now a cell array)
            add_stars = size(Gstars_temp, 2);   % Additional stars found
            for j = 1:add_stars % Allocate each star into permanent measure
                wt(Nstars + j - 1) = size(Gstars_temp{j}, 1);
                kstars{Nstars + j - 1} = kstars_temp{j};
                Gstars{Nstars + j - 1} = Gstars_temp{j};
                starIdx{Nstars + j - 1} = star_idx_temp{j};
                Tstars{Nstars + j - 1} = Tstars_temp{j};
                Gsq_stars(Nstars + j - 1) = GsqSort(i);
                symop{Nstars + j - 1} = symop_temp{j};
            end
            Nstars = Nstars + add_stars - 1;
        else % If symmetry obeyed in a star, allocate to permanent measure
            wt(Nstars) = wt_temp;
            kstars{Nstars} = kstars_temp;
            Gstars{Nstars} = Gstars_temp;
            starIdx{Nstars} = star_idx_temp;
            Tstars{Nstars} = Tstars_temp;
            Gsq_stars(Nstars) = GsqSort(i);
            symop{Nstars} = symop_temp;
        end
    end
end

%%% Start looping over waves in a star to:
%   1) Check for cancellation of stars:
%       - If waves in a star can reach the root via multiple symmetry
%       operations, we need to check that if any translation operations
%       exist in these symmetry operations, that the phases are compatible
%       (ie the ratio of phases is 1). This is the extinction condition
%       that can arise from centering and roto-translations. An easy way to
%       check this is if there exists a symmetry operation that leaves the
%       root invariant but for which its phase factor is not 1.
%   2) Closure of stars:
%       - Check that stars are closed via the *inversion operator*: these
%       stars are *closed under inversion*
%       - If stars are closed but *not* via the inversion operator: these
%       stars are *closed*
%       - If stars are not closed: these stars are *open*
%       - If stars are closed because an equivalent representation of
%       itself in the FBZ is valid, then they are closed but flag them as 1
%       in the boundary array. This prevents bad situations from happening
%       when calculating coefficients later.
%       - When the inverse root is found, label the index of the wave
%       within the star corresponding to the root and its inverse
%   3) Normalizing coefficients
%       - For stars *closed under inversion* : 1/sqrt(N_sym)
%       - For stars *closed* : figure out phase shift related to the
%       translation operator that brings the root to its opposite, then the
%       coefficient is : 1/sqrt(N_sym) * exp(-1i*phi_1/2)
%       - For stars *open* : create a summation of basis functions with a
%       coefficient of 1/sqrt(2) * 1/sqrt(N_sym) * i (if odd - see derivation)

% Initialize
cancellation = zeros(Nstars,1);
closure = zeros(Nstars,1);
coeff = zeros(Nstars,1);
link = zeros(Nstars,2);
boundary = zeros(Nstars,1);
boundary_wave = zeros(Nstars,3);
root_invroot_idx = zeros(Nstars,2);
inv_phase = zeros(Nstars, 1);
TinvPhase = cell(Nstars, 1); % Empty

% First wavevector is always closed, not cancelled, and unlinked
closure(1) = 1;
cancellation(1) = 0;
coeff(1) = 1;

for i = 2:Nstars
    %%% If star only has one wavevector, check for cancellation and nothing
    %%% else. Also set as boundary to prevent inversion problems later.
    ks_act = kstars{i};
    Gs_act = Gstars{i};
    star_waves = wt(i);
    
    %%% Stars with one wavevector can find properties immediately -- Not
    %%% true, don't skip the rest
    if wt(i) == 1
        closure(i) = 0; % What should this be?
        cancellation(i) = 0;
        %coeff(i) = 1; % Don't set this, it resets the coeff if we found a
        %link
        boundary(i) = 1;
    end

    %%% Check for cancellation of stars
    % Count the number of symmetry operations that can bring the root of
    % the star back to itself. If this number is greater than 1 (ie not
    % just the identity operator), then check that the dot product of the
    % translational component of the operator and the root is an integer
    % multiple of 2pi. In other words, the phase exp( 1i dot(root, t) )
    % must be 1. The check can be done on the ks values dotted with the
    % translation operator in fractional coordinates of the direct basis
    root = Gs_act(1,:);         % root of the star
    kroot = ks_act(1,:);
    [~, T_op, num_op, ~] = symmetry_count(kroot, param);
    if num_op > 1
        num_T = length(T_op);
        % Check only that modulus of t_op * k_star is zero
        for z = 1:num_T
            phase = dot(kroot, T_op{z});
            phase_mod = mod(phase, 1);
            if phase_mod ~= 0 + 0*1i
                cancellation(i) = 1;
                break
            end
        end
    end
    
    % Check for closure of stars. We do not need to check how the inverse
    % wavevector was arrived at. If inversion symmetry exists due to the
    % space group, then it must be due to the inversion operator. If
    % closure is achieved for a non-centrosymmetric space group, then it
    % cannot be due to an inversion operator.
    found_closed = zeros(star_waves, 1);
    for j = 1:star_waves    % Go through list of waves
        if found_closed(j)  % Already found to be closed
            continue
        else
            for k = 1:star_waves    % Check rest of waves for inverse
                if found_closed(k)  % if opposite already found
                    continue
                end
                if norm(Gs_act(j,:) + Gs_act(k,:)) <= epsilon
                    % Inverse pair found
                    found_closed(j) = 1;
                    found_closed(k) = 1;
                    % Inverse pair is of the root
                    if j == 1
                        root_invroot_idx(i,:) = [j,k];
                    end
                else
                    continue
                end
            end
        end
    end
    
    % Check if truly not closed or representation not in FBZ (on boundary)
    % This check MUST include checking the wave against itself. If an
    % inversion pair is found among the alternates with itself, then that
    % means it is an artifact of discretization (even points); we only
    % count the wavevector once. If it truly is not closed, then finding
    % all other representations shifted by M_i won't find the inverse pair.
    % Anytime we enter this loop and find a closed pair, then we flag as
    % being on the boundary. We should not be entering this loop if we are
    % not on the boundary.
    if ~all(found_closed)
        for j = 1:star_waves
            if found_closed(j)
                continue
            else
                for k = 1:star_waves
                    if found_closed(k)  % if opposite already found
                        continue
                    end
                    Galt = GbzAll(ks_act(k,:), Gbasis, Nx, Ny, Nz);
                    Gs_alt = Galt * Gbasis;
                    for l = 1:size(Galt,1)
                        if norm(Gs_act(j,:) + Gs_alt(l,:)) <= epsilon
                            % Inverse pair found with an alternate
                            % representation
                            found_closed(j) = 1;
                            found_closed(k) = 1;
                            % Inverse pair is of the root
                            if j == 1
                                root_invroot_idx(i,:) = [j,k];
                                boundary(i) = 1;
                                boundary_wave(i,:) = Gs_alt(l,:);
                            end
                        end
                    end
                end
            end
        end
    end
    closure(i) = all(found_closed);
    
    %%% Find the coefficients of all stars.
    % First check for cancellation, if cancelled, skip. If not cancelled,
    % check for one of three cases
    if cancellation(i)
        % if cancelled, set coefficient as zero (special treatment needed in
        % conversion to prevent divide by zero)
        coeff(i) = 0;
    %%% Find coefficients for uncancelled stars
    % if not cancelled, find the normalization coefficient. The phase
    % factors are found in a separate subroutine phase_find.m due to
    % the need to update those during every update of cell length
    % parameters
    else
        %%% Case 1 : Centrosymmetric and closed under inversion
        if centro && closure(i)
            % Normalization coefficient for the root
            coeff(i) = 1/sqrt(wt(i));
        %%% Case 1.5 : Technically centrosymmetric, but due to shifted
        %%% origin need to treat slightly differently. We don't want phase
        %%% modified normalization coefficients
        elseif ~centro && closure(i) && shifted_origin
            root_inv = ks_act( root_invroot_idx(i,2), : ); % We need the opposite kroot

            % Find the symmetry operator that connects the root to its
            % inverse. We should be able to take just the first T_inv
            % operator to construct the phase factor
            if boundary(i)
                root_inv = -kroot; % Override, sometimes the indexed value is wrong
            end
            [~, T_inv, ~, ~] = symmetry_operator(root_inv, kroot, boundary(i), param);
            TinvPhase{i} = T_inv(1,:);
            inv_phase(i) = exp( -1i/2 * dot(root, T_inv(1,:) * Rbasis) );
            % Reduce numerical error
            if abs(imag(inv_phase(i))) <= 1e-10
                inv_phase(i) = real(inv_phase(i));
            end
            coeff(i) = 1/sqrt(wt(i));

        %%% Case 2 : Non-centrosymmetric and closed not under inversion
        elseif ~centro && closure(i)
            % Find the phase relation between the root wavevector and its
            % inverse wavevector.
            %root_inv = Gs_act( root_invroot_idx(i,2), : );  % Use previously indexed location for the actual reciprocal wavevector (not just index)
            root_inv = ks_act( root_invroot_idx(i,2), : ); % We need the opposite kroot

            % Find the symmetry operator that connects the root to its
            % inverse. We should be able to take just the first T_inv
            % operator to construct the phase factor
            if boundary(i)
                root_inv = -kroot; % Override, sometimes the indexed value is wrong
            end
            [~, T_inv, ~, ~] = symmetry_operator(root_inv, kroot, boundary(i), param);
            TinvPhase{i} = T_inv(1,:);
            inv_phase(i) = exp( -1i/2 * dot(root, T_inv(1,:) * Rbasis) );
            % Reduce numerical error
            if abs(imag(inv_phase(i))) <= 1e-10
                inv_phase(i) = real(inv_phase(i));
            end
            coeff(i) = inv_phase(i) * 1/sqrt(wt(i));
        %%% Case 3 : Non-centrosymmetric and not closed
        elseif ~centro && ~closure(i)
            % Loop over all the stars again and check for non-closure and
            % same ||Gbz||. For all these stars, check every wavelength to
            % find the one containing the inverse wavevector. Also check
            % that the star hasn't already been sorted with its complement.
            % Also can't be the same star
            Gsq_current = Gsq_stars(i);     % Current Gsq
            found_link_flag = 0; % Reset the flag, only tripped if we find a link to the current star i
            for k = 2:Nstars    % won't be first wavevector
                if found_link_flag
                    % Exit to next star if we just found a link
                    break
                end
                Gsq_comp = Gsq_stars(k);
                if Gsq_comp == Gsq_current && ~closure(k) && all(link(k, :) == [0, 0]) && k ~= i
                    % compare all waves in the comparison star to the
                    % inverse of the root
                    comp_waves = Gstars{k};
                    % If boundary flag is up, we need to shift anything
                    % that is truly at zero back to zero, else it doesn't
                    % think the inverse is found
                    for w = 1:wt(k)
                        if boundary(k)
                            % Take alternate representation and see if any
                            % of these alternate forms indicate the correct
                            % switch
                            comp_k = kstars{k};
                            comp_k_alt = GbzAll(comp_k(w, :), param.Gbasis, param.Nx, param.Ny, param.Nz);
                            for alternate = 1:size(comp_k_alt, 1)
                                if all(-kroot == comp_k_alt(alternate, :))
                                    match_flag = 1;
                                end
                            end
                        else
                            match_flag = 0;
                        end

                        if all(-root == comp_waves(w,:)) || match_flag == 1 % inverse wavevector found either direct inverse or in the alternate reps
                            % reconstruct the complementary star to be
                            % formed from the inverse root
                            %Gs_comp = -Gs_act;  % The opposite star must contain all inverse wavevectors to the original star -- not necessarily true if not centrosymmetric. Just means the root and its inverse are reachable by all waves
                            %ks_comp = -ks_act;
                            Gs_comp = Gstars{k};
                            ks_comp = kstars{k};
                            % error checking, if the newly formed star does
                            % not match the size of the original star,
                            % return an error
                            if all(size(Gs_comp) ~= size(Gstars{k}))
                                error('Formed star containing inverse wavevector has wrong dimensions');
                            end                     
                            % Add a value to the 'link' variable which
                            % gives the linking relationship. Index it on
                            % the first arrived star which has the relation
                            link(i,:) = [i, k];
                            link(k,:) = [i, k];
                            % Overwrite the linked star to get it in the
                            % right form (inverse wavevector is root,
                            % correct set of transformations). Need to
                            % re-sort the list
                            % If the two comparisons are the same, no need
                            % to resort, just send in ks_sort and
                            % s_idx_sort exactly
                            % Always resort

                            %if 0 %sum(abs(Gs_comp - comp_waves), 'all') < 1e-12
                            %    ks_sort = kstars{k};
                            %    s_idx_sort = starIdx{k};
                            %else
                                %[ks_sort, s_idx_sort] = resort(Gs_comp, comp_waves, kstars{k}, starIdx{k}, param); % Check - Is this zeroing out waves?
                                %[ks_sort, s_idx_sort] = resort(ks_comp, kstars{k}, kstars{k}, starIdx{k}, param);
                            if w ~= 1
                                % Move the inverse of the checking wave
                                % into the first position
                                temp = Gs_comp(1, :);
                                Gs_comp(1, :) = Gs_comp(w, :);
                                Gs_comp(w, :) = temp;
                                
                                temp = ks_comp(1, :);
                                ks_comp(1, :) = ks_comp(w, :);
                                ks_comp(w, :) = temp;
                                ks_sort = ks_comp;
                                
                                s_comp = starIdx{k};
                                temp = s_comp(1, :);
                                s_comp(1, :) = s_comp(w, :);
                                s_comp(w, :) = temp;
                                s_idx_sort = s_comp;
                            else
                                ks_sort = kstars{k};
                                s_idx_sort = starIdx{k};
                            end
                            %end
                            % Find all the translation operators from waves
                            % in ks_comp to root in ks_comp
                            [~, Ts_comp, symop_comp] = is_symmetrical(ks_comp, param);
                            % Check that the phase factors between the root
                            % of the base star and the first wavevector and
                            % the root of the second star and the first
                            % wavevector in that star are the same. If they
                            % are then can directly find a new coefficient.
                            % If they aren't, then need to multiply a 1i
                            % Can only do this if there is more than 1 wave
                            % in a star
                            % Check the phases only after sorting to make
                            % sure the first wave are opposites
                            if wt(k) > 1
                                % Check if we are odd or even
                                % The criteria to fulfill is that the phase
                                % factors are equal
                                % For this check, we need to make sure
                                % Gs_act(2,:) and Gs_comp(2,:) are exactly
                                % opposites.

                                phase_base = exp(-1i * dot( Gs_act(2,:), Tstars{i}(2,:) * Rbasis ) );   % phase shift corresponding to second wavevector in base star
                                phase_comp = exp(-1i * dot( Gs_comp(2,:), Ts_comp(2,:) * Rbasis ) );    % phase shift corresponding to second wavevector in complementary star
                                % Reduce numerical error
                                if abs(imag(phase_base)) <= 1e-10
                                    phase_base = real(phase_base);
                                end
                                if abs(imag(phase_comp)) <= 1e-10
                                    phase_comp = real(phase_comp);
                                end
                                % Set coefficients
                                if phase_comp == phase_base
                                    coeff(i) = 1/sqrt(wt(i)) * 1/sqrt(2);
                                    coeff(k) = 1/sqrt(wt(k)) * 1/sqrt(2);
                                elseif phase_comp ~= phase_base
                                    coeff(i) = 1i * 1/sqrt(wt(i)) * 1/sqrt(2);
                                    coeff(k) = 1i * 1/sqrt(wt(k)) * 1/sqrt(2);
                                end
                            else
                                % Assume the only relation inside a star
                                % that works is identity, which has no
                                % phase shift
                                coeff(i) = 1/sqrt(wt(i)) * 1/sqrt(2);
                                coeff(k) = 1/sqrt(wt(k)) * 1/sqrt(2);
                            end
                            kstars{k} = ks_sort;
                            starIdx{k} = s_idx_sort;
                            Gstars{k} = Gs_comp;
                            Tstars{k} = Ts_comp;
                            symop{k} = symop_comp;
                            % Go to next star i
                            found_link_flag = 1;
                            break % Check this is working
                        else
                            continue
                        end
                    end
                else
                    continue
                end
            end
        else
            error('Unknown case, check star formation')
        end
    end
end
end

function [symmetry, Tstars, symop] = is_symmetrical(kstars_temp, param)
% Subfunction that checks if the wave being added to a star can be reached
% via group symmetry from the root wave
% NOTE: Start checking for symmetry from the 2nd group onwards since the
% first group is always the identity operator
% Check is done in CRYSTALLOGRAPHIC COORDINATES
sym = param.sym;
Rsym = param.Rsym;
Tsym = param.Tsym;
G_basis = param.Gbasis;
Nx = param.Nx;
Ny = param.Ny;
Nz = param.Nz;
epsilon = 1e-10;

Nwaves_in_star = size(kstars_temp, 1);
symmetry = zeros(Nwaves_in_star, 1);
symmetry(1) = 1;    % root is always symmetrical to itself
root = kstars_temp(1, :); % Compare current wave to root of star
Tstars(1,:) = [0, 0, 0]; % root is arbitrarily chosen to not have any translation
symop(1,:) = 1;         % the symmetry operation for the root is always the identity

for i = 2:Nwaves_in_star  % loop over all waves in star except first
    current = kstars_temp(i, :); % Active wave
    for k = 2:sym % Start checking symmetry from the second group onwards (1st is identity)
        ktrans = current * Rsym{k};     % Translation doesn't come in here
        kbztrans = GtoGbz(ktrans,G_basis,Nx,Ny,Nz,1,param);
        if norm(kbztrans - root) <= epsilon || norm(ktrans - root) <= epsilon
            % This is the operation from a wave to the root. For
            % construction of phase factors, in the phase factor function
            % we build off Gwave and not Groot, and have a negative
            symmetry(i) = 1;  % symmetry flag is set as 1 - wave found
            Tstars(i,:) = Tsym{k};    % returns which translation operation returns the wave to the root
            symop(i,:) = k;           % returns which symmetry operation was used
            break
        else
            symmetry(i) = 0;  % symmetry flag is set as 0 - wave not found
        end
    end
end
end

function [Gstars, kstars, idxstars, Tstars, symop] = deconvolute_star(Gstars_temp, kstars_temp, star_idx_temp, param)
% Subfunction that breaks apart all possible stars within the same
% magnitude Gsq if it is identified that not all waves can be reached by
% symmetry of the root

allocate = zeros(size(Gstars_temp,1), 1);   % Sets flag of whether a wave has been allocated to a star
Gactive = Gstars_temp(~allocate,:);    % waves to be tested are unallocated ones
kactive = kstars_temp(~allocate,:);
idxactive = star_idx_temp(~allocate);
[symmetry, Ts_active, symop_active] = is_symmetrical(kactive, param);  % check for all symmetrical waves with respect to root (first wave of Gactive)
allocate = symmetry;    % if symmetry found, unassign from next check (logical set to 1)
Gs_local{1} = Gactive(~~symmetry,:);     % assign the set of same waves as a single star
ks_local{1} = kactive(~~symmetry,:);
idx_local{1} = idxactive(~~symmetry);
Ts_local{1} = Ts_active(~~symmetry,:);
symop_local{1} = symop_active(~~symmetry,:);
Gactive = Gstars_temp(~allocate,:);
kactive = kstars_temp(~allocate,:);
idxactive = star_idx_temp(~allocate);

if ~all(allocate)        % if any waves still unallocated (allocate still has zero)
    [Gs_recur, ks_recur, idx_recur, Ts_recur, symop_recur] = deconvolute_star(Gactive, kactive, idxactive, param);
    Gstars = [Gs_local{1}, Gs_recur];   % automatically make a cell array of the right size
    kstars = [ks_local{1}, ks_recur];
    idxstars = [idx_local{1}, idx_recur];
    Tstars = [Ts_local{1}, Ts_recur];
    symop = [symop_local{1}, symop_recur];
else    % most nested case, assign as cell array if all allocated
    Gstars{1} = Gs_local{1};
    kstars{1} = ks_local{1};
    idxstars{1} = idx_local{1};
    Tstars = Ts_local;
    symop = symop_local;
end
end

function [R_op, T_op, num_op, idx] = symmetry_count(root, param)
% Subfunction that counts the number of symmetry operations that can bring
% the root wavevector back to itself. Returns the rotational symmetry
% operator matrix as well as the translational symmetry operator vector.
% Used to check for cancellation of stars
sym = param.sym;
Rsym = param.Rsym;
Tsym = param.Tsym;
G_basis = param.Gbasis;
Nx = param.Nx;
Ny = param.Ny;
Nz = param.Nz;
epsilon = 1e-10;
num_op = 0;

% R_op and T_op will always have an element due to symmetry operation

for k = 1:sym  % Check ALL symmetry operations (including identity)
    ktrans = root * Rsym{k};
    kbztrans = GtoGbz(ktrans,G_basis,Nx,Ny,Nz,1,param);
    if norm(kbztrans - root) <= epsilon || norm(ktrans - root) <= epsilon
        % symmetry operation works
        num_op = num_op + 1;    % Increase count of symmetry operations found
        R_op{num_op} = Rsym{k}; % Output the rotational symmetry operator
        T_op{num_op} = Tsym{k}; % Output the translational symmetry operator
        idx(num_op) = k;
    end
end
end

function [R_op, T_op, num_op, idx] = symmetry_operator(wave, root, boundary, param)
% Subfunction that lists all the symmetry operators that can bring us from
% the root to wave. Note the order of operation, it is from the root to
% wave and not the opposite. Returns the rotational symmetry
% operator matrix as well as the translational symmetry operator vector.
% Used to create coefficients for closed non-centrosymmetric stars
% If the boundary flag is active, then we should calculate equivalents at
% different edges of the FBZ because we are on a boundary. Then loop over
% all equivalents checking for symmetries
sym = param.sym;
%group = param.group;
Rsym = param.Rsym;
Tsym = param.Tsym;
G_basis = param.Gbasis;
%R_basis = param.Rbasis;
Nx = param.Nx;
Ny = param.Ny;
Nz = param.Nz;
epsilon = 1e-10;
num_op = 0;

% If we are on boundary, do the boundary check
if boundary
    for k = 1:sym  % Check symmetry operations including identity
        ktrans = root * Rsym{k};
        kbztrans = GtoGbz(ktrans,G_basis,Nx,Ny,Nz,1,param);
        % Normal check
        if norm(kbztrans - wave) <= epsilon || norm(ktrans - wave) <= epsilon
            % symmetry operation works
            num_op = num_op + 1;    % Increase count of symmetry operations found
            R_op{num_op} = Rsym{k}; % Output the rotational symmetry operator
            T_op(num_op,:) = Tsym{k}; % Output the translational symmetry operator
            idx(num_op) = k;
            continue
        end
        % Calculate if alternate forms of Gs for the current kbz satisfy
        % building off the unshifted ktrans
        kalt = GbzAll(ktrans, G_basis, Nx, Ny, Nz);
        kinv_alt = GbzAll(wave, G_basis, Nx, Ny, Nz);
        if any(ismember(kalt, kinv_alt, 'rows')) || any(ismember(kalt, wave, 'rows'))
            % symmetry operation works
            num_op = num_op + 1;
            R_op{num_op} = Rsym{k}; % Output the rotational symmetry operator
            T_op(num_op,:) = Tsym{k}; % Output the translational symmetry operator
            idx(num_op) = k;
            continue
        end

        % Calculate if alterate forms of Gs for the current kbz satisfy
        kalt = GbzAll(kbztrans, G_basis, Nx, Ny, Nz);
        if any(ismember(kalt, kinv_alt, 'rows')) || any(ismember(kalt, wave, 'rows'))
            % symmetry operation works
            num_op = num_op + 1;
            R_op{num_op} = Rsym{k}; % Output the rotational symmetry operator
            T_op(num_op,:) = Tsym{k}; % Output the translational symmetry operator
            idx(num_op) = k;
        end
    end
else
    for k = 1:sym  % Check symmetry operations including identity
        ktrans = root * Rsym{k};
        kbztrans = GtoGbz(ktrans,G_basis,Nx,Ny,Nz,1,param);
        if norm(kbztrans - wave) <= epsilon || norm(ktrans - wave) <= epsilon
            % symmetry operation works
            num_op = num_op + 1;    % Increase count of symmetry operations found
            R_op{num_op} = Rsym{k}; % Output the rotational symmetry operator
            T_op(num_op,:) = Tsym{k}; % Output the translational symmetry operator
            idx(num_op) = k;
        end
    end
end
end

function [ks_sort, s_idx_sort] = resort(Gstars_match, Gstars, kstars, star_idx, param)
% Subfunction to re-sort a star so that its order matches that of
% Gstars_match. Arrays to re-sort are Gstars, kstars, and star_idx.

% Initialize
ks_sort = zeros(size(kstars,1), size(kstars,2));
s_idx_sort = zeros(size(star_idx,1), size(star_idx,2));

% Go over every wavevector in the target star
for i = 1:size(Gstars_match,1)
    % Go over every wavevector in the star to sort
    for j = 1:size(Gstars,1)
        % If a match is found, populate the new list with it
        if all(Gstars_match(i,:) == Gstars(j,:))
            ks_sort(i,:) = kstars(j,:);
            s_idx_sort(i,:) = star_idx(j,:);
            break % Immediately exit, we found a match, move onto next i
        else
            % If a match is not found, this is because the original was not
            % inside the boundaries. We need to populate or else severe
            % problems occur later. Use GbzAll to find all representations
            % and check each one for a match
            Galt = GbzAll(Gstars(j,:), param.Gbasis, param.Nx, param.Ny, param.Nz);
            for k = 1:size(Galt,1)
                % Look for match
                if all(Gstars_match(i,:) == Galt(k,:))
                    ks_sort(i,:) = kstars(j,:);
                    s_idx_sort(i,:) = star_idx(j,:);
                end
            end
        end
    end
end
end
