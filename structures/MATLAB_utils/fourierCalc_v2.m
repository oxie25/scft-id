function [ft, fs, s_idx] = fourierCalc_v2(kstars, ptaus, numPoints, param)
%%% Oliver Xie 2022
%%% Olsen Lab - Massachusetts Institute of Technology
%%%
%%% Subfunction routine that precalculates the necessary values for
%%% conducting the crystallographic FFT and IFFT
%%%
%%% Inputs:
%%%     - kstars {Nstars, 1} : Cell array containing the k values in each
%%%     star
%%%     - ptaus {Ntaus, 1} : Cell array containing the p values in each tau
%%%     - numPoints [4, 1] : Array containing the number of discretized
%%%     points
%%%     - s_idx : Array containing only the list of uncancelled stars
%%%     - param : Struct of parameters
%%%
%%% Outputs:
%%%     - ft [Nstars, Ntau] : Array of ft(*) values
%%%     - fs [Ntau, Nstar] : Array of f*(t) values

Nstars = param.Nstars;
Ntaus = param.Ntaus;
M1 = numPoints(1);
M2 = numPoints(2);
M3 = numPoints(3);
ncoeff = param.normCoeff;
phase = param.phase;
centro = param.centro;
TinvPhase = param.TinvPhase; % used for closed but non-centrosymmetric stars
Rbasis = param.Rbasis;
Gbasis = param.Gbasis;
Gwt = param.Gwt;

nPoints = M1 * M2 * M3;

%%% We need to check if centro-symmetry is satisfied
if centro
    fs = zeros(Ntaus, Nstars);
    ft = zeros(Nstars, Ntaus);
    % Centrosymmetry satisfied, proceed normally
    
    %%% Generate a few cells and matrices that are necessary for both
    %%% calculations
    % Generate a matrix of k_root
    kRootCell = cellfun(@(x) x(1,:), kstars, 'UniformOutput', false);
    kRootMat = zeros(3, Nstars);
    
    % Fill the matrix with each vector as a column
    for i = 1:Nstars
        kRootMat(:, i) = kRootCell{i}(:); % Ensure each vector is a column
    end

    % Generate a matrix of p_root
    pRootCell = cellfun(@(x) x(1,:), ptaus, 'UniformOutput', false);
    pRootMat = zeros(3, Ntaus);
    
    % Fill the matrix with each vector as a column
    for i = 1:Ntaus
        pRootMat(:, i) = pRootCell{i}(:) ./ [M1; M2; M3]; % Ensure each vector is a column
    end
    
    % Generate the cell array for active p tau values. Each cell entry is a
    % matrix of size N x 3
    % Normalize the cell array by discretization
    pActiveCell = cellfun(@(x) x ./ [M1, M2, M3], ptaus, 'UniformOutput', false);

    % Generate the cell array for active k star values. Each cell entry is a
    % matrix of size N x 3
    kActiveCell = kstars;
    phaseActiveCell = phase;

    %%% Conditioning for FFT - from tau to star space
    % For ft, every column is calculated in the following way:
    % 1) A column has a set of taus which is found at every location
    % 1:Ntaus in pActiveCell
    % 2) This set of taus is a matrix of size N x 3. We have found another
    % matrix of size 3 x Nstars in kRootMat. We can multiply kRootMat x
    % pActiveCell{tau} to get a matrix of size N x Nstars. Transpose this
    % vector to get Nstars x N. This vector is named k_p.
    % 3) We then calculate sum(exp(-1i .* 2*pi .* k_p), 2) .* ncoeff(:) ./
    % M1 * M2 * M3) to get the column at tau.

    for tau = 1:Ntaus
        k_p = pActiveCell{tau} * kRootMat;
        k_p = k_p.'; % This is Nstars x N
        ft(:, tau) = sum(exp(-1i .* 2 .* pi .* k_p), 2) .* ncoeff(:) ./ nPoints;
    end

    %%% Conditioning for IFFT - from star to tau space
    % For fs, every column is calculated in the following way:
    % 1) A column has a set of taus which is found at every location
    % 1:Nstars in kActiveCell
    % 2) This set of stars is a matrix of size N x 3. We have found another
    % matrix of size 3 x Ntaus in pRootMat. We can multiply pRootMat x
    % kActiveCell{tau} to get a matrix of size N x Ntaus. Transpose this
    % vector to get Ntaus x N. This vector is named k_p.
    % 3) We then calculate sum(phase .* exp(1i .* 2*pi .* k_p), 2) ./ ncoeff(:) 
    % to get the column at star.
    % Only iterate through non-zero ncoeff values (or else division by 0
    % occurs)
    
    nonzerocoeff_Idx = find(ncoeff);

    for active_star_idx = 1:length(nonzerocoeff_Idx)
        star = nonzerocoeff_Idx(active_star_idx);
        k_p = kActiveCell{star} * pRootMat;
        k_p = k_p.'; % This is Ntaus x N
        fs(:, star) = sum(phaseActiveCell{star}.' .* exp(1i .* 2 .* pi .* k_p), 2) ./ ncoeff(star);
    end

else
    % Centrosymmetry not satisfied. This could be due to
    % non-centrosymmetric space group but closed star, or open star. Open
    % and closedness of a star are dependent on each star. So as we go
    % through stars, we need to check for open or closed star
    
    % Either all stars are open, or some stars are open, some are
    % closed. For open stars, we need to link the two related stars and
    % find the basis functions. For closed star with phase link, we
    % need to check the phase link.
    % We need a reduced star space, where we combine the linked
    % functions into linearly-combined basis functions satisfying the
    % reality criterion

    link = param.link;
    Nstars_mod = Nstars - nnz(link(:,1)) / 2; % Number of links, remove one count of these from Nstars
    fs_full = zeros(Ntaus, Nstars); % The full transformation matrix without removing links
    ft_full = zeros(Nstars, Ntaus); % The full transformation matrix without removing links

    %%% Generate a few cells and matrices that are necessary for both
    %%% calculations
    % Generate a matrix of k_root
    kRootCell = cellfun(@(x) x(1,:), kstars, 'UniformOutput', false);
    kRootMat = zeros(3, Nstars);
    
    % Fill the matrix with each vector as a column
    for i = 1:Nstars
        kRootMat(:, i) = kRootCell{i}(:); % Ensure each vector is a column
    end

    % Generate a matrix of p_root
    pRootCell = cellfun(@(x) x(1,:), ptaus, 'UniformOutput', false);
    pRootMat = zeros(3, Ntaus);
    
    % Fill the matrix with each vector as a column
    for i = 1:Ntaus
        pRootMat(:, i) = pRootCell{i}(:) ./ [M1; M2; M3]; % Ensure each vector is a column
    end
    
    % Generate the cell array for active p tau values. Each cell entry is a
    % matrix of size N x 3
    % Normalize the cell array by discretization
    pActiveCell = cellfun(@(x) x ./ [M1, M2, M3], ptaus, 'UniformOutput', false);

    % Generate the cell array for active k star values. Each cell entry is a
    % matrix of size N x 3
    kActiveCell = kstars;
    phaseActiveCell = phase;

    % Identify stars with the following conditions
    % Open: Link exists between it and another star
    % Imaginary coefficient: odd construction open
    % Real coefficient: even construction open
    % Tinv exists: Closed not under centrosymmetry
    % All open stars
    openList = unique(link, 'rows'); % Reduced list of unique pairs
    openList = openList(~all(openList == 0, 2), :); % Remove the 0, 0 entry
    % Find the odd and even open stars
    ncoeffImag = find(imag(ncoeff) ~= 0);
    odd_rows = ismember(openList, ncoeffImag);
    even_rows = ~ismember(openList, ncoeffImag);
    openListOdd = openList(all(odd_rows, 2), :);
    openListEven = openList(all(even_rows, 2), :);
    % Find the closed but not under centrosymmetry stars
    nonCentro = find(~cellfun('isempty', TinvPhase));

    %%% We will find the full arrays as normal, then apply modifications to
    %%% the above rows. So ft_full and fs_full are the original unmodified
    %%% entries.

    %%% Conditioning for FFT - from tau to star space
    % For ft, every column is calculated in the following way:
    % 1) A column has a set of taus which is found at every location
    % 1:Ntaus in pActiveCell
    % 2) This set of taus is a matrix of size N x 3. We have found another
    % matrix of size 3 x Nstars in kRootMat. We can multiply kRootMat x
    % pActiveCell{tau} to get a matrix of size N x Nstars. Transpose this
    % vector to get Nstars x N. This vector is named k_p.
    % 3) We then calculate sum(exp(-1i .* 2*pi .* k_p), 2) .* ncoeff(:) ./
    % M1 * M2 * M3) to get the column at tau.

    for tau = 1:Ntaus
        k_p = pActiveCell{tau} * kRootMat;
        k_p = k_p.'; % This is Nstars x N
        ft_full(:, tau) = sum(exp(-1i .* 2 .* pi .* k_p), 2) .* ncoeff(:) ./ nPoints;
    end

    % Closed stars without translation symmetry operations are calculated normally

    % Closed stars with translation symmetry operations require phase
    % factor modification
    ncoeff_nomod = 1 ./ sqrt(Gwt);

    GRootMat = kRootMat.' * Gbasis; % (Nstars x 3) x (3 x 3), select rows
    invPhase = zeros(Nstars, 1);
    invPhase(nonCentro) = exp( 1i/2 * sum(GRootMat(nonCentro, :) .* vertcat(TinvPhase{nonCentro}) * Rbasis, 2));
    ft_full(nonCentro,:) = ft_full(nonCentro,:) ./ ncoeff(nonCentro); % Un-normalize it
    ft_full(nonCentro,:) = ft_full(nonCentro,:) .* ncoeff_nomod(nonCentro).' .* invPhase(nonCentro);

    % Handle open and odd stars - subtract second from first if odd, then take
    % average
    for i = 1:size(openListOdd, 1)
        ft_full(openListOdd(i, 1), :) = 0.5 * (ft_full(openListOdd(i, 1), :) - ft_full(openListOdd(i, 2), :));
    end

    for i = 1:size(openListEven, 1)
        ft_full(openListEven(i, 1), :) = 0.5 * (ft_full(openListEven(i, 1), :) + ft_full(openListEven(i, 2), :));
    end

    % Remove the duplicate rows to form the final ft
    rows_to_remove = openList(:, 2); % all the rows to remove are the second of the open stars
    ft_full(rows_to_remove, :) = [];

    ft = ft_full;

    %%% Conditioning for IFFT - from star to tau space
    % For fs, every column is calculated in the following way:
    % 1) A column has a set of taus which is found at every location
    % 1:Nstars in kActiveCell
    % 2) This set of stars is a matrix of size N x 3. We have found another
    % matrix of size 3 x Ntaus in pRootMat. We can multiply pRootMat x
    % kActiveCell{tau} to get a matrix of size N x Ntaus. Transpose this
    % vector to get Ntaus x N. This vector is named k_p.
    % 3) We then calculate sum(phase .* exp(1i .* 2*pi .* k_p), 2) ./ ncoeff(:) 
    % to get the column at star.
    % Only iterate through non-zero ncoeff values (or else division by 0
    % occurs)
    
    nonzerocoeff_Idx = find(ncoeff);

    % We use either ncoeff_nomod directly or ncoeff to reduce numerical
    % error. Previously was doing this because we thought there was sign
    % error when transforming, but turns out we were unintentionally using
    % the conjugate transpose ' when we wanted the non-conjugate .' for
    % imaginaries.

    for active_star_idx = 1:length(nonzerocoeff_Idx)
        star = nonzerocoeff_Idx(active_star_idx);
        k_p = kActiveCell{star} * pRootMat;
        k_p = k_p.'; % This is Ntaus x N
        if ismember(star, nonCentro)
            % active star is also non centrosymmetric but unlinked.
            % Directly divide by ncoeff_nomod. Handle invPhase later
            fs_full(:, star) = sum(phaseActiveCell{star}.' .* exp(1i .* 2 .* pi .* k_p), 2) ./ ncoeff_nomod(star);
        else
            fs_full(:, star) = sum(phaseActiveCell{star}.' .* exp(1i .* 2 .* pi .* k_p), 2) ./ ncoeff(star);
        end
    end

    % Closed stars without translation symmetry operations are calculated normally

    % Closed stars with translation symmetry operations require phase
    % factor modification
    GRootMat = kRootMat.' * Gbasis; % (Nstars x 3) x (3 x 3), select rows
    invPhase = zeros(1, Nstars);
    invPhase(nonCentro) = exp( -1i/2 * sum(GRootMat(nonCentro, :) .* vertcat(TinvPhase{nonCentro}) * Rbasis, 2));
    fs_full(:, nonCentro) = fs_full(:, nonCentro) .* invPhase(nonCentro);

    % Handle open and odd stars - subtract second from first if odd, no
    % averaging
    for i = 1:size(openListOdd, 1)
        fs_full(:, openListOdd(i, 1)) = (fs_full(:, openListOdd(i, 1)) - fs_full(:, openListOdd(i, 2)));
        fs_full(:, openListOdd(i, 2)) = fs_full(:, openListOdd(i, 1)); % Set the other linked star to the same real coeff in case the wrong one is listed in rows to remove
    end

    for i = 1:size(openListEven, 1)
        fs_full(:, openListEven(i, 1)) = (fs_full(:, openListEven(i, 1)) + fs_full(:, openListEven(i, 2)));
        fs_full(:, openListEven(i, 2)) = fs_full(:, openListEven(i, 1)); % Set the other linked star to the same real coeff in case the wrong one is listed in rows to remove
    end

    % Remove the duplicate rows to form the final ft
    fs_full(:, rows_to_remove) = [];

    fs = fs_full;
end

% Find s_idx (the used stars)
s_idx = nonzerocoeff_Idx;

% Keep only real elements of ft and fs if it is below a certain tolerance.

tolerance = 1e-9;

ft(abs(imag(ft)) < tolerance) = real(ft(abs(imag(ft)) < tolerance));
fs(abs(imag(fs)) < tolerance) = real(fs(abs(imag(fs)) < tolerance));

ft(abs(ft) < 1e-12) = 0;
fs(abs(fs) < 1e-12) = 0;

end

