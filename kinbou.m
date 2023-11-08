
% テスト用のランダムな100x100の2次元配列を生成
%matrix = randi([0, 1], 100, 100);
matrix = zeros(100,100);
% matrix(50:60, 70:80) =1;
matrix(1,1) =1;
matrix(20:30, 20:30) = 1;

% 配列中の値が1である領域の重心とその近傍10か所の位置を求める
[centroids, nearby_positions] = find_centroids_and_nearby_positions(matrix);

% 結果の表示
disp('重心の座標:');
disp(centroids);

for i = 1:numel(nearby_positions)
    fprintf('領域%dの近傍10か所の座標:\n', i);
    disp(nearby_positions{i});
end

%%
% テスト用のランダムな100x100の2次元配列を生成
matrix = zeros(100,100);
matrix(50:60, 70:80) =1;
matrix(1:40,80:100) =2;
matrix(20:30, 20:30) = 3;
matrix(25:35, 25:50) = 4;

% 2次元配列中の複数の領域の境界線を描画
boundary_matrix = draw_boundaries(matrix);

% 結果の表示
figure;
imshow(boundary_matrix);
title('Boundaries of Overlapping Regions');

%%
[centroids, shrunken_matrix] = shrink_regions_around_centroids(matrix, 100);
disp(centroids);

figure;
imshow(matrix);
figure;
imshow(shrunken_matrix);

%%
Fs = 1000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 1500;             % Length of signal
t = (0:L-1)*T;        % Time vector
ff = Fs*(0:(L/2))/L;

S = zeros(2,L);
S(1,:) = 0.7*sin(2*pi*50*t) + sin(2*pi*120*t);
Y = fft(S');
Y = Y';
P2 = abs(Y/L);
P1 = P2(:,1:L/2+1);
P1(:,2:end-1) = 2*P1(:,2:end-1);

figure;
plot(ff,P1(1,:)) 
title("Single-Sided Amplitude Spectrum of S(t)")
xlabel("f (Hz)")
ylabel("|P1(f)|")

%%

VmaxPk = 2;       % Maximum operating voltage
Fi = 2000;        % Sinusoidal frequency of 2 kHz
Fs = 44.1e3;      % Sample rate of 44.1kHz
Tstop = 50e-3;    % Duration of sinusoid
t = 0:1/Fs:Tstop; % Input time vector

% Use the maximum allowable voltage of the amplifier
inputVmax = VmaxPk*sin(2*pi*Fi*t);
outputVmax = helperHarmonicDistortionAmplifier(inputVmax);

%%
Fs = 1000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 1500;             % Length of signal
t = (0:L-1)*T;        % Time vector

S = 0.7*sin(2*pi*50*t) + sin(2*pi*120*t);
X = S + 2*randn(size(t));

Y = fft(X);

P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = Fs*(0:(L/2))/L;
plot(f,P1) 
title("Single-Sided Amplitude Spectrum of X(t)")
xlabel("f (Hz)")
ylabel("|P1(f)|")

%%
function [centroids, nearby_positions] = find_centroids_and_nearby_positions(matrix)
    % Step 1: Find connected regions using bwconncomp
    connected_regions = bwconncomp(matrix);

    % Step 2: Calculate centroids for each connected region
    stats = regionprops(connected_regions, 'Centroid');
    centroids = cat(1, stats.Centroid);

    % Step 3: Find nearby positions for each centroid
    nearby_positions = cell(size(centroids, 1), 1);
    for i = 1:size(centroids, 1)
        cx = round(centroids(i, 1));
        cy = round(centroids(i, 2));

        % Extract nearby positions within a 10x10 neighborhood
        xmin = max(1, cx - 5);
        xmax = min(size(matrix, 2), cx + 5);
        ymin = max(1, cy - 5);
        ymax = min(size(matrix, 1), cy + 5);

        % Create a 10x10 sub-matrix for nearby positions
        nearby_submatrix = matrix(ymin:ymax, xmin:xmax);

        % Find the indices of 1's within the sub-matrix
        [ny, nx] = find(nearby_submatrix);

        % Calculate the actual positions relative to the whole matrix
        nearby_positions{i} = [ny + (ymin - 1), nx + (xmin - 1)];
    end
end

%%
function boundary_matrix = draw_boundaries(matrix)
    % Step 1: Find connected regions using bwconncomp
    connected_regions = bwconncomp(matrix);

    % Initialize boundary matrix with zeros
    boundary_matrix = zeros(size(matrix));

    % Step 2: Identify boundaries for each connected region
    for region_idx = 1:connected_regions.NumObjects
        % Extract the current region
        region_mask = zeros(size(matrix));
        region_mask(connected_regions.PixelIdxList{region_idx}) = 1;

        % Use MATLAB function to find boundaries of the region
        region_boundaries = bwboundaries(region_mask, 'noholes');

        % Step 3: Draw boundaries on the boundary matrix
        boundary_points = region_boundaries{1};
        for point_idx = 1:size(boundary_points, 1)
            boundary_x = boundary_points(point_idx, 2);
            boundary_y = boundary_points(point_idx, 1);

            % Set boundary points to 1 in the boundary matrix
            boundary_matrix(boundary_y, boundary_x) = 1;
        end
    end
end

%%
function [centroids, shrunken_matrix] = shrink_regions_and_combine(matrix, shrink_ratio)
    % Step 1: Find connected regions using bwconncomp
    connected_regions = bwconncomp(matrix);

    % Initialize the shrunken matrix with zeros
    shrunken_matrix = zeros(size(matrix));

    % Step 2: Calculate the centroids of the regions
    stats = regionprops(connected_regions, 'Centroid');
    centroids = cat(1, stats.Centroid);

    % Step 3: Shrink each region around its centroid and combine in the shrunken matrix
    for region_idx = 1:connected_regions.NumObjects
        % Extract the current region
        region_mask = zeros(size(matrix));
        region_mask(connected_regions.PixelIdxList{region_idx}) = 1;

        % Calculate the displacement vector from the centroid to each pixel
        [yy, xx] = find(region_mask);
        displacement = [xx - centroids(region_idx, 1), yy - centroids(region_idx, 2)];

        % Scale the displacement vector according to the shrink ratio
        scaled_displacement = displacement * shrink_ratio;

        % Calculate the new pixel coordinates after scaling
        new_xx = round(centroids(region_idx, 1) + scaled_displacement(:, 1));
        new_yy = round(centroids(region_idx, 2) + scaled_displacement(:, 2));

        % Ensure that the new coordinates are within the matrix bounds
        new_xx = max(1, min(size(matrix, 2), new_xx));
        new_yy = max(1, min(size(matrix, 1), new_yy));

        % Set the pixels of the shrunken region to 1 in the shrunken matrix
        shrunken_matrix(sub2ind(size(matrix), new_yy, new_xx)) = 1;
    end
end

%%

function [centroids, shrunken_matrix] = shrink_regions_around_centroids(matrix, n)
    % Step 1: Find connected regions using bwconncomp
    connected_regions = bwconncomp(matrix);

    % Initialize the 3D logical array to store shrunken regions
    shrunken_regions = false(size(matrix, 1), size(matrix, 2), connected_regions.NumObjects);

    % Step 2: Calculate the centroids of the regions
    stats = regionprops(connected_regions, 'Centroid');
    centroids = cat(1, stats.Centroid);

    % Step 3: Shrink each region around its centroid and select n closest positions
    for region_idx = 1:connected_regions.NumObjects
        % Extract the current region
        region_mask = zeros(size(matrix));
        region_mask(connected_regions.PixelIdxList{region_idx}) = 1;

        % Calculate the distance from the centroid of the region to all pixels in the region
        [yy, xx] = find(region_mask);
        distances = sqrt((yy - centroids(region_idx, 2)).^2 + (xx - centroids(region_idx, 1)).^2);

        % Sort the distances in ascending order
        [sorted_distances, idx] = sort(distances);

        % Select the n closest positions and their corresponding coordinates
        selected_indices = idx(1:min(n, length(idx)));
        selected_positions = [xx(selected_indices), yy(selected_indices)];

        % Create the shrunken region for the current region
        shrunken_region = false(size(matrix));
        shrunken_region(sub2ind(size(matrix), selected_positions(:, 2), selected_positions(:, 1))) = 1;

        % Store the shrunken region in the 3D logical array
        shrunken_regions(:, :, region_idx) = shrunken_region;
    end

    % Combine the 3D logical array into a single logical array
    shrunken_matrix = any(shrunken_regions, 3);
end







