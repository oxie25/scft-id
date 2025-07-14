function [R, T, centro, shifted_origin, inversion_translation] = symExtract(space_group, dim, targetFolder)
% Extract the matrices and vectors for performing symmetry operations. Use
% the symmetry files for space groups in the Morse paper
% If the inversion matrix is NOT one of the operators, then there is no
% centrosymmetry

if dim == 1
    subfolder = '1';
elseif dim == 2
    subfolder = '2';
elseif dim == 3
    subfolder = '3';
else
    error('Invalid value for dim. Choose 1, 2, or 3.');
end

% Construct full path to the file in the target folder and subfolder
fullPath = fullfile(targetFolder, subfolder, space_group);

% Open the file
fid = fopen(fullPath, 'r');
if fid == -1
    error('File not found: %s', fullPath);
end

% Read the first two lines
header1 = fgetl(fid);
header2 = fgetl(fid);

% Read the rest of the file as strings to handle fractions
fileContents = textscan(fid, '%s', 'Delimiter', '\n', 'CollectOutput', true);
fclose(fid);

% Extract the data lines as a cell array of strings
dataLines = fileContents{1};
if isempty(dataLines{1})
    dataLines(1) = []; % Remove the first blank line
numLines = length(dataLines);

% Initialize cell array to store 4x3 matrices
R = {};
T = {};
centro = 0; % Raise as true if inversion matrix found
rowIdx = 1;
shifted_origin = 0;
inversion_translation = [0, 0, 0];

% Inversion matrix
inversionMatrix = -eye(dim);

% Every matrix has size dim+1 x dim
while rowIdx <= numLines
    matrix = zeros(dim+1, dim); % Pre-allocate matrix
    for i = 0:dim  % Read dim lines
        line = strtrim(dataLines{rowIdx + i});
        values = regexp(line, '\s+', 'split');  % Split by one or more spaces
        
        % Convert each value (handle fractions with eval)
        for j = 1:dim
            if contains(values{j}, '/')
                matrix(i + 1, j) = eval(values{j}); % Evaluate fraction (e.g., '3/4' -> 0.75)
            else
                matrix(i + 1, j) = str2double(values{j}); % Convert decimal directly
            end
        end
    end
    
    % Split off the last row as the translation vector
    rotation = zeros(3);
    translation = zeros(1, 3);

    rotation(1:dim, 1:dim) = matrix(1:dim, :);
    translation(1:dim) = matrix(end,:);
    
    % Store
    R{end + 1} = rotation;  % Rotation matrix
    T{end + 1} = translation; % Translation vector

    % Check for centrosymmetry (presence of inversion) and no glide - 
    % if glide exists it means the origin (0, 0, 0) is not at center of inversion
    if (isequal(matrix(1:dim, :), inversionMatrix)) && (~any(matrix(end, :)))
        centro = 1;  % Set centro flag to 1
    end

    if (isequal(matrix(1:dim, :), inversionMatrix)) && (any(matrix(end, :)))
        shifted_origin = 1;  % This means that the inversion operator exists, but the origin is shifted. We consider this as not centrosymmetric for calculation purposes but in reality it is
        inversion_translation = translation; % This is the translation operator of the inversion operation
    end

    % There is a blank line blank line
    rowIdx = rowIdx + (dim+2);  % Move to the next set of matrices

end
end