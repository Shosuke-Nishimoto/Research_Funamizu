[X,cmap] = imread("D:\Nishimoto_data\ISOI\230923_ISOI_S6_1\230923_ISOI_S6_1_tif\230923_ISOI_S6_1_X1.tif");
X = imresize(X, 1/8);
X = ind2rgb(X,cmap);
figure;
imshow(X)
%%
imds = imageDatastore("D:\Nishimoto_data\ISOI\230923_ISOI_S6_1\230923_ISOI_S6_1_tif");
im_list = imds.Files;
ii = regexp(im_list,'(?<=X)\d*','match');
[i0,i0] = sort(str2double([ii{:}]));
file_name = im_list(i0);

raw_data = [];
strlen =  0;  % 表示する文字列の長さを記録する変数
iteNum = size(file_name,1);  % 適当に繰り返す試行数
fprintf('Pleese wait .........\n');
h = waitbar(0,'Please wait...');
s = clock;

warning('off','all');
for i = 1:iteNum
    Tmp = {'Current trial: %3d/%d\n', i, iteNum};
    Tmp{1} = [ repmat(sprintf('\b'),[1 strlen]),  Tmp{1} ];

    Txt = sprintf(Tmp{1:3});
    fprintf(Txt);
    strlen = length(Txt) - strlen;

    temp = imread(file_name{i});
    raw_data(:,:,i) = imresize(temp, 1/8);
%     raw_data(:,:,i) = temp(55:230,125:305);

    % begin estimate remaining time
    if i ==1
        is = etime(clock,s);
        esttime = is * iteNum;
    end
    h = waitbar(i/iteNum,h,...
        ['remaining time =',num2str(esttime-etime(clock,s),'%4.1f'),'sec' ]);
    % end estimate remaining time
end
close(gcf);
warning('on','all');
warning('query','all')

imshow(uint8(raw_data(:,:,5)))

%%
save_filename = "230923_ISOI_S6_1";
save(save_filename+".mat",'raw_data','iteNum',"-v7.3");

% 512*512のまま、目的の領野だけとってきて、画素数を落とさずに解析

%% import data
clearvars

[X,cmap] = imread("D:\Nishimoto_data\ISOI\230923_ISOI_S6_2\230923_ISOI_S6_2_tif\230923_ISOI_S6_2_X1.tif");
X = imresize(X, 1/8);
X = ind2rgb(X,cmap);

load('230923_ISOI_S6_2.mat')
load('NI_23-Sep-2023_1732_02.bin.mat');

figure;
plot(ch(1,:))
hold on;
plot(ch(7,:))
hold off;

%% z-scoreで汎用的に使えるようにしてもいいかも
% カメラの撮影タイミング
num_images = [];
for i = 1:length(ch)-1
    if ch(1,i+1)>=3.9 && ch(1,i)<3.9 % ここはセットにより変わる
        num_images(end+1) = i+1;
    end
end
% num_images = num_images(32:end);

% 点滅刺激の提示タイミング
num_stimuli = [];
for i = 1:length(ch)-1
    if ch(7,i+1)>=0.2 && ch(7,i)<0.2 % ここはセットにより変わる
        num_stimuli(end+1) = i+1;
    end
end

% 刺激一枚一枚を提示したタイミングの推定値 60Hzに変換
% 最後だけ内分をとらずに推定値で埋める
% 30Hzに変換
estim_num_stimuli = [];
estim_num_stimuli(1) = num_stimuli(1);
for i = 1:length(num_stimuli)
    if i < length(num_stimuli)
        naibun = round(linspace(num_stimuli(i),num_stimuli(i+1),6));
        estim_num_stimuli(5*i-3) = naibun(2);
        estim_num_stimuli(5*i-2) = naibun(3);
        estim_num_stimuli(5*i-1) = naibun(4);
        estim_num_stimuli(5*i) = naibun(5);
        estim_num_stimuli(5*i+1) = naibun(6);
    else
        naibun = round(linspace(num_stimuli(i-1),num_stimuli(i),6))-num_stimuli(i-1);
        estim_num_stimuli(5*i-3) = num_stimuli(i)+naibun(2);
        estim_num_stimuli(5*i-2) = num_stimuli(i)+naibun(3);
        estim_num_stimuli(5*i-1) = num_stimuli(i)+naibun(4);
        estim_num_stimuli(5*i) = num_stimuli(i)+naibun(5);
    end
end

%%

% カメラのタイミングとイメージング画像のindex set
num_images(2,:) = 1:length(num_images);

% 刺激のタイミングと提示刺激のindex set
max_stim_num = 750; % 30Hzだから1500の半分
estim_num_stimuli(2,1:length(estim_num_stimuli)/2) = mod(0:(length(estim_num_stimuli)/2-1), max_stim_num) + 1;
estim_num_stimuli(2,length(estim_num_stimuli)/2+1:length(estim_num_stimuli))...
                                    = mod(0:(length(estim_num_stimuli)/2-1), max_stim_num) + max_stim_num + 1;

% イメージング画像のindex と提示刺激のindexをalign
align_matrix = closest_search(num_images,estim_num_stimuli);
image_stim_align = [];
for i = 1:size(align_matrix,2)/max_stim_num
    if align_matrix(4,max_stim_num*i)-align_matrix(4,max_stim_num*(i-1)+1)==max_stim_num-1
        image_stim_align = cat(2,image_stim_align,align_matrix(:,max_stim_num*(i-1)+1:max_stim_num*i));
    end
end
image_stim_align = image_stim_align([1 4],:);

bound = find(image_stim_align(1,:)==max_stim_num+1);
is_align_azimuth = image_stim_align(:,bound(1):end);
is_align_altitude = image_stim_align(:,1:bound(1)-1);

%% データをピクセル*時間に変換
data = raw_data;
clearvars raw_data

image_data = zeros(size(data,1)*size(data,2),size(data,3));
for i = 1:size(data,3)
    image_data(:,i) = reshape(data(:,:,i),size(data,1)*size(data,2),1);
end

%% altitude, azimuthの時の平均活動に変換

image_data_azimuth = image_data(:, is_align_azimuth(2,:));
del_image_data_azimuth = image_data_azimuth;
damy = zeros(size(image_data_azimuth,1), max_stim_num);
repeat_num_azimuth = size(image_data_azimuth,2)/max_stim_num;
for i = 1:repeat_num_azimuth
    f = mean(image_data_azimuth(:,max_stim_num*(i-1)+1:max_stim_num*(i-1)+61),2);
    del_f = (image_data_azimuth(:,max_stim_num*(i-1)+1:max_stim_num*i)-f)./f;
    del_image_data_azimuth(:,max_stim_num*(i-1)+1:max_stim_num*i) = del_f;
    damy = damy + del_f;
end
av_image_data_azimuth = damy./repeat_num_azimuth;

image_data_altitude = image_data(:, is_align_altitude(2,:));
del_image_data_altitude = image_data_altitude;
damy = zeros(size(image_data_altitude,1), max_stim_num);
repeat_num_altitude = size(image_data_altitude,2)/max_stim_num;
for i = 1:repeat_num_altitude
    f = mean(image_data_altitude(:,max_stim_num*(i-1)+1:max_stim_num*(i-1)+61),2);
    del_f = (image_data_altitude(:,max_stim_num*(i-1)+1:max_stim_num*i)-f)./f;
    del_image_data_altitude(:,max_stim_num*(i-1)+1:max_stim_num*i) = del_f;
    damy = damy + del_f;
end
av_image_data_altitude = damy./repeat_num_altitude;

%%
figure;
plot(1:max_stim_num, av_image_data_azimuth(:,:))
figure;
plot(1:max_stim_num, av_image_data_altitude(:,:))

%% 平均で計算
Fs = 30; % Sampling frequency
T = 1/Fs; % Sampling period
L = size(av_image_data_altitude,2); % Length of signal
t = (0:L-1)*T; % Time vector
ff = Fs*(0:(L/2))/L;
% フーリエ変換を実行
fourier_altitude = fft(av_image_data_altitude');
fourier_azimuth = fft(av_image_data_azimuth');

% 第1高調波成分を抽出
first_harmonic_idx = 2; % 1/25 Hz
first_harmonic_altitude = fourier_altitude(first_harmonic_idx,:);
first_harmonic_azimuth = fourier_azimuth(first_harmonic_idx,:);
% 第1高調波成分の位相を抽出
phase_angle_altitude = angle(first_harmonic_altitude);
phase_angle_altitude = reshape(phase_angle_altitude,size(X,1),size(X,2));
phase_angle_azimuth = angle(first_harmonic_azimuth);
phase_angle_azimuth = reshape(phase_angle_azimuth,size(X,1),size(X,2));

%%
% 配列のサイズを取得
[num_rows, num_cols] = size(phase_angle_altitude);
% x座標とy座標のグリッドを生成
[grid_x, grid_y] = meshgrid(1:num_cols, 1:num_rows);

figure;
imshow(X);
hold on;
scatter(grid_x(:), grid_y(:), 1, phase_angle_altitude(:), 'filled');
colorbar
colormap('jet')
% clim([-pi -1])
hold off;

figure;
imshow(X);
hold on;
scatter(grid_x(:), grid_y(:), 1, phase_angle_azimuth(:), 'filled');
colorbar
colormap('jet')
% clim([pi/2 pi])
hold off;

%%
[Gmag_altitude,Gdir_altitude] = imgradient(phase_angle_altitude);
[Gmag_azimuth,Gdir_azimuth] = imgradient(phase_angle_azimuth);

sign_map = sin((Gdir_altitude-Gdir_azimuth)*pi/180);
% SD = 0.5;
% sign_map(sign_map>SD) =1; sign_map(sign_map<-SD) = -1; sign_map(sign_map<=SD & sign_map>=-SD) =0;
figure;
imshow(X);
hold on;
scatter(grid_x(:), grid_y(:), 1, sign_map(:), 'filled');
colorbar
colormap('jet')
hold off;
%%
function [centroids, shrunken_matrix] = shrink_regions_around_centroids(matrix, n)
    % Step 1: Find connected regions using bwconncomp
    filtered_matrix = bwareaopen(matrix, 10);
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

function align_matrix = closest_search(a,b)
    align_matrix = zeros(4,size(b,2));
    align_matrix(1,:) = b(2,:);
    align_matrix(2,:) = b(1,:);
    for i = 1:size(b,2)
        d = a(1,:)-b(1,i);
        d(d<0) = 1000;
        [M,I] = min(d);
        align_matrix(3,i) = a(1,I);
        align_matrix(4,i) = I;
    end
end














