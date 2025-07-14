function [ybz, GsqBz] = GtoGbz(y, Gbasis, Nx, Ny, Nz, mode, param)
%%% Oliver Xie 2021
%%% Olsen Lab - Massachusetts Institute of Technology
%%%
%%% Subfunction routine to shift all wavevectors into the First Brillouin
%%% Zone (FBZ). This is done by translating each wavevector by the periodic
%%% translation operator (a full domain spacing in every reciprocal basis
%%% direction). The representation with the lowest Gbz^2 value from the
%%% origin is the FBZ representation. If wavevectors are on the boundary,
%%% then the representation with positive i integer (then j, then k) is
%%% returned. Also returns the FBZ-shifted square magnitude (since it needs
%%% to be calculated anyways)
%%%
%%% Inputs:
%%%     - y [1 x 3] : Either the wavevector in Cartesian coordinates OR the
%%%     wavevector index
%%%     - Gbasis [3 x 3] : The reciprocal basis vector in Cartesian
%%%     coordinates
%%%     - Nx, Ny, Nz : Number of points in the unit cell
%%%     - mode : Tells program to return either the wavevectors in
%%%     Cartesian representation in the FBZ or the wavevector indices of
%%%     of the wavevector in the FBZ. Mode 1 returns the wavevector indices
%%%     while Mode 2 returns the wavevector
%%%     - param : Struct of parameters
%%%
%%% Outputs:
%%%     - ybz : Either the FBZ wavevector in Cartesian coordinates or the
%%%     FBZ wavevector index depending on the mode
%%%     - GsqBz : Square magnitudes of each wavevector after shifting into
%%%     the FBZ

% Initialize
GsqMin = 1E10;
dim = param.dim;

ymin = y;

switch mode
    case 1  % Returns wavevector indices
        switch dim
            case 1
                for i1 = 1:-1:-1
                    ytry(1) = y(1) + i1*Nx;
                    ytry(2) = y(2);
                    ytry(3) = y(3);
                    % Construct new wavevector upon shift
                    for i = 1:1
                        v(i) = ytry(:)' * Gbasis(:,i);
                    end
                    % Calculate the new magnitude
                    Gsq = v * v(:);
                    % Compare the new magnitudes, keep if less than
                    % current, reject if not.
                    if Gsq < GsqMin
                        ymin = ytry;
                        GsqMin = Gsq;
                    end
                end
            case 2
                for i1 = 1:-1:-1
                    ytry(1) = y(1) + i1*Nx;
                    for i2 = 1:-1:-1
                        ytry(2) = y(2) + i2*Ny;
                        ytry(3) = y(3);
                        % Construct new wavevector upon shift
                        for i = 1:2
                            v(i) = ytry(:)' * Gbasis(:,i);
                        end
                        % Calculate the new magnitude
                        Gsq = v * v(:);
                        % Compare the new magnitudes, keep if less than
                        % current, reject if not.
                        if Gsq < GsqMin
                            ymin = ytry;
                            GsqMin = Gsq;
                        end
                    end
                end
            case 3
                for i1 = 1:-1:-1
                    ytry(1) = y(1) + i1*Nx;
                    for i2 = 1:-1:-1
                        ytry(2) = y(2) + i2*Ny;
                        for i3 = 1:-1:-1
                            ytry(3) = y(3) + i3*Nz;
                            % Construct new wavevector upon shift
                            for i = 1:3
                                v(i) = ytry(:)' * Gbasis(:,i);
                            end
                            % Calculate the new magnitude
                            Gsq = v * v(:);
                            % Compare the new magnitudes, keep if less than
                            % current, reject if not.
                            if Gsq < GsqMin
                                ymin = ytry;
                                GsqMin = Gsq;
                            end
                        end
                    end
                end
        end
        
        
    case 2 % Returns wavevector
        switch dim
            case 1
                for i1 = 1:-1:-1
                    % Equivalent to dot product, resulting vector is
                    % written in Cartesian coordinate [x y z]
                    ytry = y + (Nx*i1 * Gbasis(1,:));
                    v = ytry;
                    % Calculate the new magnitude
                    Gsq = v * v(:);
                    % Compare the new magnitudes, keep if less than
                    % current, reject if not.
                    if Gsq < GsqMin
                        ymin = ytry;
                        GsqMin = Gsq;
                    end
                end
            case 2
                for i1 = 1:-1:-1
                    for i2 = 1:-1:-1
                        % Equivalent to dot product, resulting vector is
                        % written in Cartesian coordinate [x y z]
                        ytry = y + (Nx*i1 * Gbasis(1,:)) + (Ny*i2 * Gbasis(2,:));
                        v = ytry;
                        % Calculate the new magnitude
                        Gsq = v * v(:);
                        % Compare the new magnitudes, keep if less than
                        % current, reject if not.
                        if Gsq < GsqMin
                            ymin = ytry;
                            GsqMin = Gsq;
                        end
                    end
                end
            case 3
                for i1 = 1:-1:-1
                    for i2 = 1:-1:-1
                        for i3 = 1:-1:-1
                            % Equivalent to dot product, resulting vector is
                            % written in Cartesian coordinate [x y z]
                            ytry = y + (Nx*i1 * Gbasis(1,:)) + (Ny*i2 * Gbasis(2,:)) + (Nz*i3 * Gbasis(3,:));
                            v = ytry;
                            % Calculate the new magnitude
                            Gsq = v * v(:);
                            % Compare the new magnitudes, keep if less than
                            % current, reject if not.
                            if Gsq < GsqMin
                                ymin = ytry;
                                GsqMin = Gsq;
                            end
                        end
                    end
                end
        end
end

% Assign output
ybz = ymin;
GsqBz = GsqMin;

end