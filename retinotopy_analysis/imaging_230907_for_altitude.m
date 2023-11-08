% [X,cmap] = imread("C:\Users\funam\Desktop\analysis\imaging\230808_Blue_tiff\230808_Blue_X1.tif");
% X = X(55:230,125:305);
% X = ind2rgb(X,cmap);
% figure;
% imshow(X)
%%
% imds = imageDatastore("C:\Users\funam\Desktop\analysis\imaging\230808_Blue_tiff");
% imds = imageDatastore("C:\Users\funam\Desktop\analysis\imaging\230905_in_bar_2_tiff");
% im_list = imds.Files;
% ii = regexp(im_list,'(?<=X)\d*','match');
% [i0,i0] = sort(str2double([ii{:}]));
% file_name = im_list(i0);
% 
% raw_data = [];
% strlen =  0;  % 表示する文字列の長さを記録する変数
% iteNum = size(file_name,1);  % 適当に繰り返す試行数
% fprintf('Pleese wait .........\n');
% h = waitbar(0,'Please wait...');
% s = clock;
% 
% warning('off','all');
% for i = 1:iteNum
%     Tmp = {'Current trial: %3d/%d\n', i, iteNum};
%     Tmp{1} = [ repmat(sprintf('\b'),[1 strlen]),  Tmp{1} ];
% 
%     Txt = sprintf(Tmp{1:3});
%     fprintf(Txt);
%     strlen = length(Txt) - strlen;
% 
%     temp = imread(file_name{i});
%     %raw_data(:,:,i) = imresize(temp, 1/4);
%     raw_data(:,:,i) = temp(55:230,125:305);
% 
%     % begin estimate remaining time
%     if i ==1
%         is = etime(clock,s);
%         esttime = is * iteNum;
%     end
%     h = waitbar(i/iteNum,h,...
%         ['remaining time =',num2str(esttime-etime(clock,s),'%4.1f'),'sec' ]);
%     % end estimate remaining time
% end
% close(gcf);
% warning('on','all');
% warning('query','all')
% 
% imshow(uint8(raw_data(:,:,5)))

%%
% save_filename = "230905_in_bar_2";
% save(save_filename+".mat",'raw_data','iteNum',"-v7.3");

% 512*512のまま、目的の領野だけとってきて、画素数を落とさずに解析

%% import data
load('230912_ISOI_2.mat')
load('NI_12-Sep-2023_2109_09.bin.mat');

%%

Fs = 60; % sampling frequency in Hz
dt = 1/Fs; % time step in seconds
T = length(raw_data); % number of time points
t = (0:T-1)*dt; % time vector

%%
figure;
plot(ch(1,:))
figure;
plot(ch(7,:))

%% z-scoreで汎用的に使えるようにしてもいいかも
% カメラの撮影タイミング
num_images = [];
for i = 1:length(ch)-1
    if ch(1,i+1)>=4 && ch(1,i)<4 % ここはセットにより変わる
        num_images(end+1) = i+1;
    end
end

% 点滅刺激の提示タイミング
num_stimuli = [];
for i = 1:length(ch)-1
    if ch(7,i+1)>=0.2 && ch(7,i)<0.2 % ここはセットにより変わる
        num_stimuli(end+1) = i+1;
    end
end
% 最初のシグナルがとれてなかった場合
num_stimuli = [num_stimuli(1)-167 num_stimuli];

% 刺激一枚一枚を提示したタイミングの推定値
% 最後だけ内分をとらずに推定値で埋める
estim_num_stimuli = [];
estim_num_stimuli(1) = num_stimuli(1);
for i = 1:length(num_stimuli)
    if i < length(num_stimuli)
        naibun = round(linspace(num_stimuli(i),num_stimuli(i+1),11));
        estim_num_stimuli(10*i-8) = naibun(2);
        estim_num_stimuli(10*i-7) = naibun(3);
        estim_num_stimuli(10*i-6) = naibun(4);
        estim_num_stimuli(10*i-5) = naibun(5);
        estim_num_stimuli(10*i-4) = naibun(6);
        estim_num_stimuli(10*i-3) = naibun(7);
        estim_num_stimuli(10*i-2) = naibun(8);
        estim_num_stimuli(10*i-1) = naibun(9);
        estim_num_stimuli(10*i) = naibun(10);
        estim_num_stimuli(10*i+1) = naibun(11);
    else
        naibun = round(linspace(num_stimuli(i-1),num_stimuli(i),11))-num_stimuli(i-1);
        estim_num_stimuli(10*i-8) = num_stimuli(i)+naibun(2);
        estim_num_stimuli(10*i-7) = num_stimuli(i)+naibun(3);
        estim_num_stimuli(10*i-6) = num_stimuli(i)+naibun(4);
        estim_num_stimuli(10*i-5) = num_stimuli(i)+naibun(5);
        estim_num_stimuli(10*i-4) = num_stimuli(i)+naibun(6);
        estim_num_stimuli(10*i-3) = num_stimuli(i)+naibun(7);
        estim_num_stimuli(10*i-2) = num_stimuli(i)+naibun(8);
        estim_num_stimuli(10*i-1) = num_stimuli(i)+naibun(9);
        estim_num_stimuli(10*i) = num_stimuli(i)+naibun(10);
    end
end

%%

% カメラのタイミングとイメージング画像のindex set
num_images(2,:) = 1:length(num_images);

% 刺激のタイミングと提示刺激のindex set
% max_stim_num = 900;
max_stim_num = 1200;
estim_num_stimuli(2,1:length(estim_num_stimuli)/2) = mod(0:(length(estim_num_stimuli)/2-1), max_stim_num) + 1;
estim_num_stimuli(2,length(estim_num_stimuli)/2+1:length(estim_num_stimuli))...
                                    = mod(0:(length(estim_num_stimuli)/2-1), max_stim_num) + max_stim_num + 1;

% イメージング画像のindex と提示刺激のindexをalign
align_point = knnsearch(num_images(1,:)', estim_num_stimuli(1,1));
align_length = min(length(estim_num_stimuli), length(num_images)-align_point+1);

image_stim_align(1,:) = estim_num_stimuli(2,1:align_length);
image_stim_align(2,:) = num_images(2,align_point:align_point+align_length-1);

%% 4を超えるところと0.2を超えるところをresting stateとした
% df/fを計算
rest_duration = find(ch(7,:,:)>0.2, 1 ) - find(ch(1,:,:)>4, 1 );
rest_data = raw_data(:,:,1:image_stim_align(2,1)-1); % 改変230807
f = mean(rest_data,3);
data = (raw_data-f)./f;

%% データをピクセル*時間に変換
image_data = zeros(size(data,1)*size(data,2),size(data,3));
for i = 1:size(data,3)
    image_data(:,i) = reshape(data(:,:,i),size(data,1)*size(data,2),1);
end

% rest_dataをピクセル*時間に変換
rest_data_new = (rest_data-f)./f;
rest_image_data = zeros(size(rest_data,1)*size(rest_data,2),size(rest_data,3));
for i = 1:size(rest_data,3)
    rest_image_data(:,i) = reshape(rest_data_new(:,:,i),size(rest_data,1)*size(rest_data,2),1);
end

%% altitude, azimuthの時の平均活動に変換

% is_align_azimuth = image_stim_align(:,length(estim_num_stimuli)/2+1:length(estim_num_stimuli));
% is_align_altitude = image_stim_align(:,1:length(estim_num_stimuli)/2);

% inverse アリの場合
stim_frames = length(estim_num_stimuli) - 2*max_stim_num;
is_align_azimuth = image_stim_align(:,stim_frames/2+1:stim_frames);
is_align_altitude = image_stim_align(:,1:stim_frames/2);
is_align_inv_azimuth = image_stim_align(:,stim_frames+max_stim_num+1:length(estim_num_stimuli));
is_align_inv_altitude = image_stim_align(:,stim_frames+1:stim_frames+max_stim_num);

image_data_azimuth = image_data(:, is_align_azimuth(2,:));
image_data_inv_azimuth = image_data(:, is_align_inv_azimuth(2,-(1:end-1)+end));
image_data_inv_azimuth(:,end+1) = image_data(:, is_align_inv_azimuth(2,end));

damy = zeros(size(image_data_azimuth,1), max_stim_num);
repeat_num = size(image_data_azimuth,2)/max_stim_num;
for i = 1:repeat_num
    damy = damy + image_data_azimuth(:,max_stim_num*(i-1)+1:max_stim_num*i);
end
av_image_data_azimuth = damy./repeat_num;
image_data_inv_azimuth ...
    = image_data_inv_azimuth(mod(0:size(image_data_inv_azimuth,2)*repeat_num-1, size(image_data_inv_azimuth,2))+1);

image_data_altitude = image_data(:, is_align_altitude(2,:));
image_data_inv_altitude = image_data(:, is_align_inv_altitude(2,-(1:end-1)+end));
image_data_inv_altitude(:,end+1) = image_data(:, is_align_inv_altitude(2,end));

damy = zeros(size(image_data_altitude,1), max_stim_num);
repeat_num = size(image_data_altitude,2)/max_stim_num;
for i = 1:repeat_num
    damy = damy + image_data_altitude(:,max_stim_num*(i-1)+1:max_stim_num*i);
end
av_image_data_altitude = damy./repeat_num;
image_data_inv_altitude ...
    = image_data_inv_altitude(mod(0:size(image_data_inv_altitude,2)*repeat_num-1, size(image_data_inv_altitude,2))+1);

%%
% figure;
% plot(1:max_stim_num, av_image_data_azimuth(:,:))
% figure;
% plot(1:max_stim_num, av_image_data_altitude(:,:))

image_data_azimuth = (image_data_azimuth+image_data_inv_azimuth)/2;
image_data_altitude = (image_data_altitude+image_data_inv_altitude)/2;

%%
% 4つの刺激方向のそれぞれについて、刺激トリガーによる平均△Fムービーを作成し、
% 各方向で10～40回の試行を平均した。方位と高度の位置マップを作成するため、
% 各ピクセルの蛍光対時間データから、フーリエ系列の第一高調波成分の位相から網膜視床位置を抽出した。

[X,cmap] = imread("C:\Users\funam\Desktop\analysis\imaging\230808_Blue_tiff\230808_Blue_X1.tif");
X = X(55:230,125:305);
X = ind2rgb(X,cmap);

% altitudeもazimuthも386/900で0度の位置に到達
% 0,0の点を取ってきてその重心を計算
zero_point = 386;

% max-min >= 0.08のみをとってくる
selected_index = [];
for i = 1:size(av_image_data_azimuth,1)
    if max(av_image_data_azimuth(i,:)) - min(av_image_data_azimuth(i,:)) >= 0.08
        selected_index(end+1) = i;
    end
end

% max of azimuth
azi_diff = mean(av_image_data_azimuth(:,zero_point+31:zero_point+91),2)...
            - mean(av_image_data_azimuth(:,zero_point-59:zero_point),2);
[sorted_array, sorted_indices] = sort(azi_diff);
azi_largest_indices = sorted_indices(end-9999:end);
azimuth_center_low = sorted_indices(end);
azi_largest_indices = intersect(azi_largest_indices,selected_index);
% max of altitude
alt_diff = mean(av_image_data_altitude(:,zero_point+31:zero_point+91),2)...
            - mean(av_image_data_altitude(:,zero_point-59:zero_point),2);
[sorted_array, sorted_indices] = sort(alt_diff);
alt_largest_indices = sorted_indices(end-9999:end);
altitude_center_low = sorted_indices(end-100);
alt_largest_indices = intersect(alt_largest_indices,selected_index);
% max of all
largest_indices = intersect(azi_largest_indices,alt_largest_indices);
temp = zeros(size(image_data,1),1);
temp(largest_indices) = 1;
temp = reshape(temp,size(data,1),size(data,2));
[centroids, shrunken_matrix] = shrink_regions_around_centroids(temp, 100);
centroids = round(centroids);
% 1つ目の要素がx座標（列）で2つ目が行
center = size(data,1)*(centroids(1,1)-1)+centroids(1,2);

figure;
plot(1:900, av_image_data_azimuth(center,:))
figure;
plot(1:900, av_image_data_altitude(center,:))

%% rest_dataをピクセル*時間に変換
rest_data_new = (rest_data-f)./f;
rest_image_data = zeros(size(rest_data,1)*size(rest_data,2),size(rest_data,3));
for i = 1:size(rest_data,3)
    rest_image_data(:,i) = reshape(rest_data_new(:,:,i),size(rest_data,1)*size(rest_data,2),1);
end

% 相互相関関数
% 相互相関と位相ズレの計算 どこを基準にとるかで大きく変わる
cross_corr_rest = zeros(size(rest_image_data,1),1);
for i = 1:size(rest_image_data,1)
    temp = xcorr(rest_image_data(center,:), rest_image_data(i,:),'coef');
    cross_corr_rest(i) = temp((length(temp)+1)/2);
end
cross_corr_rest = reshape(cross_corr_rest,size(rest_data,1),size(rest_data,2));

% ピアソン相関係数
% cross_corr_rest = zeros(size(rest_image_data,1),1);
% for i = 1:size(rest_image_data,1)
%     temp = corrcoef(rest_image_data(center,:), rest_image_data(i,:)); % ピアソン相関 線形
%     cross_corr_rest(i) = temp(2,1);
% end
% cross_corr_rest = reshape(cross_corr_rest,size(rest_data,1),size(rest_data,2));

%% scatterでかっこよく表示

% 配列のサイズを取得
[num_rows, num_cols] = size(cross_corr_rest);

% x座標とy座標のグリッドを生成
[grid_x, grid_y] = meshgrid(1:num_cols, 1:num_rows);

figure;
imshow(X);
hold on;
scatter(grid_x(:), grid_y(:), 2, cross_corr_rest(:), 'filled');
colorbar
colormap("hsv")
clim([0.89 0.9])
hold off;

% マップを見て閾値を決定
vis_lim = 0.9;
%%

clearvars raw_data 

Fs = 60; % Sampling frequency
T = 1/Fs; % Sampling period
L = size(image_data_altitude,2); % Length of signal
t = (0:L-1)*T; % Time vector
ff = Fs*(0:(L/2))/L;
% フーリエ変換を実行
fourier_altitude = fft(image_data_altitude');
fourier_azimuth = fft(image_data_azimuth');

% 第1高調波成分を抽出
first_harmonic_idx = 13; % 1/20 Hz
first_harmonic_altitude = fourier_altitude(first_harmonic_idx,:);
first_harmonic_azimuth = fourier_azimuth(first_harmonic_idx,:);
% 第1高調波成分の位相を抽出
phase_angle_altitude = angle(first_harmonic_altitude);
phase_angle_altitude = reshape(phase_angle_altitude,size(data,1),size(data,2));
phase_angle_azimuth = angle(first_harmonic_azimuth);
phase_angle_azimuth = reshape(phase_angle_azimuth,size(data,1),size(data,2));

%%
[X,cmap] = imread("C:\Users\funam\Desktop\analysis\imaging\230808_Blue_tiff\230808_Blue_X1.tif");
X = X(55:230,125:305);
X = ind2rgb(X,cmap);
% 配列のサイズを取得
[num_rows, num_cols] = size(phase_angle_altitude);
% x座標とy座標のグリッドを生成
[grid_x, grid_y] = meshgrid(1:num_cols, 1:num_rows);

figure;
imshow(X);
hold on;
scatter(grid_x(:), grid_y(:), 2, phase_angle_altitude(:), 'filled');
colorbar
colormap('jet')
hold off;

figure;
imshow(X);
hold on;
scatter(grid_x(:), grid_y(:), 2, phase_angle_azimuth(:), 'filled');
colorbar
colormap('jet')
hold off;

figure;
imshow(X);
hold on;
temp = phase_angle_altitude;
temp(cross_corr_rest<vis_lim) = 100;
scatter(grid_x(:), grid_y(:), 2, temp(:), 'filled');
colorbar
colormap('jet')
clim([-2.8 -2])
hold off;

figure;
imshow(X);
hold on;
temp = phase_angle_azimuth;
temp(cross_corr_rest<vis_lim) = -100;
scatter(grid_x(:), grid_y(:), 2, temp(:), 'filled');
colorbar
colormap('jet')
clim([0 3])
hold off;

%%

[Gmag_altitude,Gdir_altitude] = imgradient(phase_angle_altitude);
[Gmag_azimuth,Gdir_azimuth] = imgradient(phase_angle_azimuth);

sign_map = sin(Gdir_altitude-Gdir_azimuth);

figure;
imshow(X);
hold on;
scatter(grid_x(:), grid_y(:), 2, sign_map(:), 'filled');
colorbar
colormap('jet')
hold off;

%% altitudeは0(20)度から、azimuthは50度から始める
% altitude -30~0, azimuth -40~40

alt_point_low = 386; %  0
alt_point_high = 258; % -30
azi_point_low = 600; % 50
azi_point_high = 172; % -50

% max of azimuth
% low
azi_diff = mean(av_image_data_azimuth(:,azi_point_low+31:azi_point_low+91),2)...
            - mean(av_image_data_azimuth(:,azi_point_low-59:azi_point_low),2);
[sorted_array, sorted_indices] = sort(azi_diff);
azi_largest_indices = sorted_indices(end-9999:end);
azi_largest_indices = intersect(azi_largest_indices,selected_index);
temp = zeros(size(image_data,1),1);
temp(azi_largest_indices) = 1;
temp(cross_corr_rest<vis_lim) = 0;
temp = reshape(temp,size(data,1),size(data,2));
[azi_centroids_low, shrunken_matrix] = shrink_regions_around_centroids(temp, 100);
azi_centroids_low = round(azi_centroids_low);
azimuth_center_low = size(data,1)*(azi_centroids_low(1,1)-1)+azi_centroids_low(1,2);
% high
azi_diff = mean(av_image_data_azimuth(:,azi_point_high+31:azi_point_high+91),2)...
            - mean(av_image_data_azimuth(:,azi_point_high-59:azi_point_high),2);
[sorted_array, sorted_indices] = sort(azi_diff);
azi_largest_indices = sorted_indices(end-9999:end);
azi_largest_indices = intersect(azi_largest_indices,selected_index);
temp = zeros(size(image_data,1),1);
temp(azi_largest_indices) = 1;
temp(cross_corr_rest<vis_lim) = 0;
temp = reshape(temp,size(data,1),size(data,2));
[azi_centroids_high, shrunken_matrix] = shrink_regions_around_centroids(temp, 100);
azi_centroids_high = round(azi_centroids_high);
azimuth_center_high = size(data,1)*(azi_centroids_high(1,1)-1)+azi_centroids_high(1,2);

% max of altitude
% low
alt_diff = mean(av_image_data_altitude(:,alt_point_low+31:alt_point_low+91),2)...
            - mean(av_image_data_altitude(:,alt_point_low-59:alt_point_low),2);
[sorted_array, sorted_indices] = sort(alt_diff);
alt_largest_indices = sorted_indices(end-9999:end);
alt_largest_indices = intersect(alt_largest_indices,selected_index);
temp = zeros(size(image_data,1),1);
temp(alt_largest_indices) = 1;
temp(cross_corr_rest<vis_lim) = 0;
temp = reshape(temp,size(data,1),size(data,2));
[alt_centroids_low, shrunken_matrix] = shrink_regions_around_centroids(temp, 100);
alt_centroids_low = round(alt_centroids_low);
altitude_center_low = size(data,1)*(alt_centroids_low(1,1)-1)+alt_centroids_low(1,2);
% high
alt_diff = mean(av_image_data_altitude(:,alt_point_high+31:alt_point_high+91),2)...
            - mean(av_image_data_altitude(:,alt_point_high-59:alt_point_high),2);
[sorted_array, sorted_indices] = sort(alt_diff);
alt_largest_indices = sorted_indices(end-9999:end);
alt_largest_indices = intersect(alt_largest_indices,selected_index);
temp = zeros(size(image_data,1),1);
temp(alt_largest_indices) = 1;
temp(cross_corr_rest<vis_lim) = 0;
temp = reshape(temp,size(data,1),size(data,2));
[alt_centroids_high, shrunken_matrix] = shrink_regions_around_centroids(temp, 100);
alt_centroids_high = round(alt_centroids_high);
altitude_center_high = size(data,1)*(alt_centroids_high(1,1)-1)+alt_centroids_high(1,2);

figure;
hold on;
plot((1:900)/60, av_image_data_azimuth(azimuth_center_low,:))
plot((1:900)/60, av_image_data_azimuth(azimuth_center_high,:))
hold off;
figure;
hold on;
plot((1:900)/60, av_image_data_altitude(altitude_center_low,:))
plot((1:900)/60, av_image_data_altitude(altitude_center_high,:))
hold off;

alt_lim = [-30 0];
azi_lim = [-50 50];
%%

angle_altitude = translate_to_angle(phase_angle_altitude, alt_centroids_high, alt_centroids_low, alt_lim);
[Gmag_altitude,Gdir_altitude] = imgradient(angle_altitude);
angle_azimuth = translate_to_angle(phase_angle_azimuth, azi_centroids_high, azi_centroids_low, azi_lim);
[Gmag_azimuth,Gdir_azimuth] = imgradient(angle_azimuth);

sign_map = sin((Gdir_altitude-Gdir_azimuth)*pi/180);

figure;
imshow(X);
hold on;
scatter(grid_x(:), grid_y(:), 2, angle_altitude(:), 'filled');
colorbar
colormap('jet')
hold off;

figure;
imshow(X);
hold on;
scatter(grid_x(:), grid_y(:), 2, angle_azimuth(:), 'filled');
colorbar
colormap('jet')
hold off;

figure;
imshow(X);
hold on;
scatter(grid_x(:), grid_y(:), 2, sign_map(:), 'filled');
colorbar
colormap('jet')
hold off;

%%

figure;
imshow(X);
hold on;
contour(phase_angle_altitude,20);
scatter(grid_x(:), grid_y(:), 2, henka_altitude(:), 'filled');
colorbar
colormap('jet')
hold off;

%%
function [centroids, shrunken_matrix] = shrink_regions_around_centroids(matrix, n)
    % Step 1: Find connected regions using bwconncomp
    filtered_matrix = bwareaopen(matrix, 20);
    connected_regions = bwconncomp(filtered_matrix);

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

function angle = translate_to_angle(phase, centroid_high, centroid_low, angle_lim)
    angle = phase - phase(centroid_high(2),centroid_high(1));
    angle = angle/angle(centroid_low(2),centroid_low(1))* (angle_lim(2)-angle_lim(1)) + angle_lim(1);
end