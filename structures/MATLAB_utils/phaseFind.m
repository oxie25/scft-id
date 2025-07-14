function phase = phaseFind(Gbasis, Rbasis, coeff, Tstars, kstars, Nstars)
%%% Oliver Xie 2021
%%% Olsen Lab - Massachusetts Institute of Technology
%%%
%%% Subfunction routine to construct the phase factors associated with each
%%% star which is multiplied to the Fourier coefficient of the root
%%% wavevector to construct the Fourier coefficient of all other
%%% wavevectors in the star
%%%
%%% Inputs:
%%%     - Gbasis : Reciprocal basis vector
%%%     - Rbasis : Direct space basis vector
%%%     - coeff : Normalization coefficient of the star
%%%     - Tstars : The set of translational symmetry operators associated
%%%     with each star
%%%     - kstars : The wavevector indices of each star
%%%     - Nstars : The number of stars
%%%
%%% Outputs:
%%%     - phase : Cell array of phase factors returned as a vector

phase = cell(Nstars,1);

for star = 1:Nstars
    % transforming to each wavevector from root in a given star
    if coeff(star) == 0
        % star is cancelled, skip this iteration
        phase{star} = 0;
    else
        % star is not cancelled
        Twave = Tstars{star};
        kwave = kstars{star};
        % need to calculate updated list of Gwave due to change in basis
        % (length optimization)
        Gwave = zeros(size(kwave,1),size(kwave,2));
        for row = 1:size(kwave,1)
            Gwave(row,:) = kwave(row,:) * Gbasis;
        end
        % phase_shift is the vector of phase shifts for each star
        phase_shift = zeros(size(Twave,1),1);
        for t = 1:size(Twave,1)
            phase_shift(t) = exp(-1i * dot( Gwave(t,:), Twave(t,:) * Rbasis ) );   % phase shift corresponding to each wave in star
            % Reduce numerical error
            if abs(imag(phase_shift(t))) <= 1e-10
                phase_shift(t) = real(phase_shift(t));
            end
        end
        phase{star} = phase_shift;
    end
end
end

