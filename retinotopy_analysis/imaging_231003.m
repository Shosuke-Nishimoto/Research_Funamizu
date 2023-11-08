clearvars
[filename, pathname]=uigetfile('*.tif');

[X,cmap] = imread(append(pathname,filename));
X = imresize(X, 1/8);
% X = X(121:195,121:215);
X = ind2rgb(X,cmap);
figure;
imshow(X)

%%
imds = imageDatastore(pathname);
im_list = imds.Files;
ii = regexp(im_list,'(?<=X)\d*','match');
[i0,i0] = sort(str2double([ii{:}]));
file_name = im_list(i0);


iteNum = size(file_name,1);  % 適当に繰り返す試行数
raw_data = zeros(size(X,1),size(X,2),iteNum);

% WaitMessage = parfor_wait(iteNum, 'Waitbar', true);
parfor i = 1:iteNum
    temp = imread(file_name{i});
    raw_data(:,:,i) = imresize(temp, 1/8);
%     temp = imresize(temp, 1/8);
%     raw_data(:,:,i) = temp(121:195,121:215);
%     WaitMessage.Send;
end
% WaitMessage.Destroy

%%
save_filename = "231008_S1_2";
save(save_filename+".mat",'raw_data','iteNum',"-v7.3");

%% import data forward
clearvars

[filename, pathname]=uigetfile('*.tif');
[X,cmap] = imread(append(pathname,filename));
X = imresize(X, 1/8);
% X = X(121:195,121:215);
X = ind2rgb(X,cmap);

load("E:\Nishimoto_ISI\231008_S1_2\231008_S1_2.mat")
load("E:\Nishimoto_ISI\231008_S1_2\NI_08-Oct-2023_2017_09.bin.mat");

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
estim_num_stimuli(2,1:length(estim_num_stimuli)/4) = mod(0:(length(estim_num_stimuli)/4-1), max_stim_num) + 1;
estim_num_stimuli(2,length(estim_num_stimuli)/4+1:length(estim_num_stimuli)/2)...
    = mod(0:(length(estim_num_stimuli)/4-1), max_stim_num) + max_stim_num + 1;
estim_num_stimuli(2,length(estim_num_stimuli)/2+1:3*length(estim_num_stimuli)/4)...
    = mod(0:(length(estim_num_stimuli)/4-1), max_stim_num) + 2*max_stim_num + 1;
estim_num_stimuli(2,3*length(estim_num_stimuli)/4+1:length(estim_num_stimuli))...
    = mod(0:(length(estim_num_stimuli)/4-1), max_stim_num) + 3*max_stim_num + 1;


% altitude, azimuthの最初の一回は解析に入れない
estim_num_stimuli = estim_num_stimuli(:,[max_stim_num+1:length(estim_num_stimuli)/4 length(estim_num_stimuli)/4+max_stim_num+1:length(estim_num_stimuli)/2 ...
                                            length(estim_num_stimuli)/2+max_stim_num+1:3*length(estim_num_stimuli)/4 ...
                                            3*length(estim_num_stimuli)/4+max_stim_num+1:end]);

% イメージング画像のindex と提示刺激のindexをalign
align_matrix = closest_search(num_images,estim_num_stimuli);
image_stim_align = [];
for i = 1:size(align_matrix,2)/max_stim_num
    if align_matrix(4,max_stim_num*i)-align_matrix(4,max_stim_num*(i-1)+1)==max_stim_num-1
        image_stim_align = cat(2,image_stim_align,align_matrix(:,max_stim_num*(i-1)+1:max_stim_num*i));
    end
end
image_stim_align = image_stim_align([1 4],:);

bound_1 = find(image_stim_align(1,:)==max_stim_num+1);
bound_2 = find(image_stim_align(1,:)==2*max_stim_num+1);
bound_3 = find(image_stim_align(1,:)==3*max_stim_num+1);
is_align_azimuth_f = image_stim_align(:,bound_1(1):bound_2(1)-1);
is_align_altitude_f = image_stim_align(:,1:bound_1(1)-1);
is_align_azimuth_b = image_stim_align(:,bound_3(1):end);
is_align_altitude_b = image_stim_align(:,bound_2(1):bound_3(1)-1);

%% データをピクセル*時間に変換
data = raw_data;
clearvars raw_data

image_data = zeros(size(data,1)*size(data,2),size(data,3));
for i = 1:size(data,3)
    image_data(:,i) = reshape(data(:,:,i),size(data,1)*size(data,2),1);
end

%% altitude, azimuthの時の平均活動に変換

image_data_azimuth_f = image_data(:, is_align_azimuth_f(2,:));
damy = zeros(size(image_data_azimuth_f,1), max_stim_num);
repeat_num_azimuth_f = size(image_data_azimuth_f,2)/max_stim_num;
for i = 1:repeat_num_azimuth_f
    f = mean(image_data_azimuth_f(:,max_stim_num*(i-1)+1:max_stim_num*(i-1)+61),2);
    del_f = (image_data_azimuth_f(:,max_stim_num*(i-1)+1:max_stim_num*i)-f)./f;
    damy = damy + del_f;
end
av_image_data_azimuth_f = damy./repeat_num_azimuth_f;

image_data_altitude_f = image_data(:, is_align_altitude_f(2,:));
damy = zeros(size(image_data_altitude_f,1), max_stim_num);
repeat_num_altitude_f = size(image_data_altitude_f,2)/max_stim_num;
for i = 1:repeat_num_altitude_f
    f = mean(image_data_altitude_f(:,max_stim_num*(i-1)+1:max_stim_num*(i-1)+61),2);
    del_f = (image_data_altitude_f(:,max_stim_num*(i-1)+1:max_stim_num*i)-f)./f;
    damy = damy + del_f;
end
av_image_data_altitude_f = damy./repeat_num_altitude_f;

image_data_azimuth_b = image_data(:, is_align_azimuth_b(2,:));
damy = zeros(size(image_data_azimuth_b,1), max_stim_num);
repeat_num_azimuth_b = size(image_data_azimuth_b,2)/max_stim_num;
for i = 1:repeat_num_azimuth_b
    f = mean(image_data_azimuth_b(:,max_stim_num*(i-1)+151:max_stim_num*(i-1)+211),2);
    del_f = (image_data_azimuth_b(:,max_stim_num*(i-1)+1:max_stim_num*i)-f)./f;
    damy = damy + del_f;
end
av_image_data_azimuth_b = damy./repeat_num_azimuth_b;

image_data_altitude_b = image_data(:, is_align_altitude_b(2,:));
damy = zeros(size(image_data_altitude_b,1), max_stim_num);
repeat_num_altitude_b = size(image_data_altitude_b,2)/max_stim_num;
for i = 1:repeat_num_altitude_b
    f = mean(image_data_altitude_b(:,max_stim_num*(i-1)+151:max_stim_num*(i-1)+211),2);
    del_f = (image_data_altitude_b(:,max_stim_num*(i-1)+1:max_stim_num*i)-f)./f;
    damy = damy + del_f;
end
av_image_data_altitude_b = damy./repeat_num_altitude_b;

%%
figure;
plot(1:max_stim_num, av_image_data_azimuth_f(:,:))
figure;
plot(1:max_stim_num, av_image_data_altitude_f(:,:))
figure;
plot(1:max_stim_num, av_image_data_azimuth_b(:,:))
figure;
plot(1:max_stim_num, av_image_data_altitude_b(:,:))

%% 平均で計算
Fs = 30; % Sampling frequency
T = 1/Fs; % Sampling period
L = size(av_image_data_altitude_f,2); % Length of signal
t = (0:L-1)*T; % Time vector
ff = Fs*(0:(L/2))/L;
% フーリエ変換を実行
fourier_altitude_f = fft(av_image_data_altitude_f');
fourier_azimuth_f = fft(av_image_data_azimuth_f');
fourier_altitude_b = fft(av_image_data_altitude_b');
fourier_azimuth_b = fft(av_image_data_azimuth_b');

% 第1高調波成分を抽出
first_harmonic_idx = 2; % 1/25 Hz
first_harmonic_altitude_f = fourier_altitude_f(first_harmonic_idx,:);
first_harmonic_azimuth_f = fourier_azimuth_f(first_harmonic_idx,:);
first_harmonic_altitude_b = fourier_altitude_b(first_harmonic_idx,:);
first_harmonic_azimuth_b = fourier_azimuth_b(first_harmonic_idx,:);
% 第1高調波成分の位相を抽出
phase_angle_altitude_f = angle(first_harmonic_altitude_f);
phase_angle_altitude_f = reshape(phase_angle_altitude_f,size(X,1),size(X,2));
phase_angle_azimuth_f = angle(first_harmonic_azimuth_f);
phase_angle_azimuth_f = reshape(phase_angle_azimuth_f,size(X,1),size(X,2));
phase_angle_altitude_b = angle(first_harmonic_altitude_b);
phase_angle_altitude_b = reshape(phase_angle_altitude_b,size(X,1),size(X,2));
phase_angle_azimuth_b = angle(first_harmonic_azimuth_b);
phase_angle_azimuth_b = reshape(phase_angle_azimuth_b,size(X,1),size(X,2));

%%
% 配列のサイズを取得
[num_rows, num_cols] = size(phase_angle_altitude_f);
% x座標とy座標のグリッドを生成
[grid_x, grid_y] = meshgrid(1:num_cols, 1:num_rows);

figure;
imshow(X);
hold on;
scatter(grid_x(:), grid_y(:), 1, phase_angle_altitude_f(:), 'filled');
colorbar
colormap('jet')
% clim([-pi -pi/2])
hold off;

figure;
imshow(X);
hold on;
scatter(grid_x(:), grid_y(:), 1, phase_angle_azimuth_f(:), 'filled');
colorbar
colormap('jet')
% clim([-pi -pi/2])
hold off;

figure;
imshow(X);
hold on;
scatter(grid_x(:), grid_y(:), 1, phase_angle_altitude_b(:), 'filled');
colorbar
colormap('jet')
% clim([pi/2 pi])
hold off;

figure;
imshow(X);
hold on;
scatter(grid_x(:), grid_y(:), 1, phase_angle_azimuth_b(:), 'filled');
colorbar
colormap('jet')
% clim([pi/2 pi])
hold off;

%%
alt_angle = translate_to_abs_angle(phase_angle_altitude_f, phase_angle_altitude_b);
azi_angle = translate_to_abs_angle(phase_angle_azimuth_f, phase_angle_azimuth_b);

figure;
imshow(X);
hold on;
scatter(grid_x(:), grid_y(:), 1, alt_angle(:), 'filled');
colorbar
colormap('jet')
clim([-40 60])
hold off;

figure;
imshow(X);
hold on;
scatter(grid_x(:), grid_y(:), 1, azi_angle(:), 'filled');
colorbar
colormap('jet')
clim([-60 60])
hold off;

%%
[Gmag_altitude,Gdir_altitude] = imgradient(phase_angle_altitude_f);
[Gmag_azimuth,Gdir_azimuth] = imgradient(phase_angle_azimuth_f);

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
function angle = translate_to_abs_angle(phase_f, phase_b)
    stim_start = 0; stim_end = 2*pi*20/25;
    a = phase_f; b = phase_b;
    d = find(a>-(stim_start+8*2*pi/25)); % ヘモダイナミクスは6sくらいなのでゆとりをもって8sとして8*2*pi/25とした
    a(d) = a(d)-2*pi;
    d = find(b>-(2*pi-stim_end+8*2*pi/25)); % ヘモダイナミクスは6sくらいなのでゆとりをもって8sとして8*2*pi/25とした
    b(d) = b(d)-2*pi;
    angle = (b-a)/2+pi;
    angle = angle*180/pi*(5/8); % 180/(360*4/5)
    angle = angle - 90;
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

function disp_progress(~)
    disp(p)
    p = p+1;
end