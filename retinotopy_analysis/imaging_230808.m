% [X,cmap] = imread("C:\Users\funam\Desktop\analysis\imaging\230808_Blue_tiff\230808_Blue_X1.tif");
% X = X(55:230,125:305);
% X = ind2rgb(X,cmap);
% figure;
% imshow(X)
%%
% imds = imageDatastore("C:\Users\funam\Desktop\analysis\imaging\230808_Blue_tiff");
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
% save_filename = "230808_Blue";
% save(save_filename+".mat",'raw_data','iteNum',"-v7.3");

% 512*512のまま、目的の領野だけとってきて、画素数を落とさずに解析

%%
x_0 = 15*1920/70.71; % 32インチの横幅70.71センチ分が1920
[y,z] = meshgrid(-960:960, -540:540);
theta = acos(z./sqrt(x_0^2 + y.^2 + z.^2))-pi/2;
phi = atan(y/x_0);
fs = 25*pi/180;
%%

s_1 = cos(2*pi*fs.*theta);
s_2 = cos(2*pi*fs.*phi);

s_1 = abs(s_1);
s_2 = abs(s_2);

% 25 degree きざみ
ind_small_1 = (abs(theta)>=0 & abs(theta)<12.5/180*pi) | (abs(theta)>=37.5/180*pi & abs(theta)<62.5/180*pi)...
    | (abs(theta)>=87.5/180*pi);
ind_large_1 = (abs(theta)>=12.5/180*pi & abs(theta)<37.5/180*pi) | (abs(theta)>=62.5/180*pi & abs(theta)<87.5/180*pi);
ind_small_2 = (abs(phi)>=0 & abs(phi)<12.5/180*pi) | (abs(phi)>=37.5/180*pi & abs(phi)<62.5/180*pi)...
    | (abs(phi)>=87.5/180*pi);
ind_large_2 = (abs(phi)>=12.5/180*pi & abs(phi)<37.5/180*pi) | (abs(phi)>=62.5/180*pi & abs(phi)<87.5/180*pi);

s_1(ind_small_1) = 0;
s_1(ind_large_1) = 1;
s_2(ind_small_2) = 0;
s_2(ind_large_2) = -1;

im = abs(s_1 + s_2);

figure;
imshow(im)

%% altitudeにnumbering
% G,R,Bの順に動かす

altitude_num = zeros(size(y,1), size(y,2), 18);
altitude_num_RGB = zeros(size(y,1), size(y,2), 3, 18);
colour_2 = zeros(18,3);
colour_2(15,:) = [1 0 0];
colour_2(14,:) = [1 0.5 0];
colour_2(13,:) = [1 1 0];
colour_2(12,:) = [0.5 1 0];
colour_2(11,:) = [0 1 0];
colour_2(10,:) = [0 1 0.5];
colour_2(9,:) = [0 1 1];
colour_2(8,:) = [0 0.5 1];
colour_2(7,:) = [0 0 1];
colour_2(6,:) = [0.5 0 1];
colour_2(5,:) = [1 0 1];
colour_2(4,:) = [1 0 0.5];

for i = 1:size(colour_2,1)
    ind = theta >= (90-10*i)/180*pi & theta < (90-10*(i-1))/180*pi;
    temp = zeros(size(y,1), size(y,2));
    temp_R = temp; temp_G = temp; temp_B = temp;
    temp(ind) = 1;
    temp_R(ind) = colour_2(i,1);
    temp_G(ind) = colour_2(i,2);
    temp_B(ind) = colour_2(i,3);
    altitude_num(:,:,i) = temp;
    altitude_num_RGB(:,:,:,i) = cat(3,temp_R,temp_G,temp_B);
end

figure;
imshow(sum(altitude_num_RGB,4));

%% azimuthにnumbering
% G,R,Bの順に動かす

azimuth_num = zeros(size(y,1), size(y,2), 16);
azimuth_num_RGB = zeros(size(y,1), size(y,2), 3, 16);
colour_1 = zeros(16,3);
colour_1(3,:) = [1 0 0];
colour_1(4,:) = [1 0.5 0];
colour_1(5,:) = [1 1 0];
colour_1(6,:) = [0.5 1 0];
colour_1(7,:) = [0 1 0];
colour_1(8,:) = [0 1 0.5];
colour_1(9,:) = [0 1 1];
colour_1(10,:) = [0 0.5 1];
colour_1(11,:) = [0 0 1];
colour_1(12,:) = [0.5 0 1];
colour_1(13,:) = [1 0 1];
colour_1(14,:) = [1 0 0.5];

for i = 1:size(colour_1,1)
    ind = phi >= (-90+180/16*(i-1))/180*pi & phi < (-90+180/16*i)/180*pi;
    temp = zeros(size(y,1), size(y,2));
    temp_R = temp; temp_G = temp; temp_B = temp;
    temp(ind) = 1;
    temp_R(ind) = colour_1(i,1);
    temp_G(ind) = colour_1(i,2);
    temp_B(ind) = colour_1(i,3);
    azimuth_num(:,:,i) = temp;
    azimuth_num_RGB(:,:,:,i) = cat(3,temp_R,temp_G,temp_B);
end

figure;
imshow(sum(azimuth_num_RGB,4));

%% import data
load('230808_Blue.mat')
load('NI_08-Aug-2023_1519_35.bin.mat');

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
max_stim_num = 900;
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

%% 1-9のaltitude, azimuthに刺激が出ていたかどうか

% altitude_judge = zeros(size(altitude_num,3),max_stim_num);
% for i = 1:max_stim_num
%     temp = load("stim_"+num2str(i));
%     temp = temp.n;
%     temp(1:50, end-50:end) = 0;
%     for ii = 1:size(altitude_num,3)
%         ttemp = temp+altitude_num(:,:,ii);
%         altitude_judge(ii,i) = abs(isempty(find(ttemp==2, 1))-1);
%     end
% end
% 
% azimuth_judge = zeros(size(azimuth_num,3),max_stim_num);
% for i = 1:max_stim_num
%     temp = load("stim_"+num2str(i+max_stim_num));
%     temp = temp.n;
%     temp(1:50, end-50:end) = 0;
%     for ii = 1:size(azimuth_num,3)
%         ttemp = temp+azimuth_num(:,:,ii);
%         azimuth_judge(ii,i) = abs(isempty(find(ttemp==2, 1))-1);
%     end
% end
% 
% save_filename = "altitude_azimuth_judge";
% save(save_filename+".mat",'altitude_judge','azimuth_judge',"-v7.3");

load('altitude_azimuth_judge.mat')

%% 実際に提示した刺激の1-9のaltitude, azimuthに刺激があったかどうか
% altitudeをやってからazimuthをやっている

altitude_judge_real = altitude_judge;
azimuth_judge_real = azimuth_judge;
q = length(image_stim_align)/(2*max_stim_num);

is_align_azimuth = image_stim_align(:,length(estim_num_stimuli)/2+1:length(estim_num_stimuli));
is_align_altitude = image_stim_align(:,1:length(estim_num_stimuli)/2);

for i = 1:q-1
    altitude_judge_real = cat(2,altitude_judge_real,altitude_judge);
    azimuth_judge_real = cat(2,azimuth_judge_real,azimuth_judge);
end

%% データをピクセル*時間に変換
image_data = zeros(size(data,1)*size(data,2),size(data,3));
for i = 1:size(data,3)
    image_data(:,i) = reshape(data(:,:,i),size(data,1)*size(data,2),1);
end

%% altitude, azimuthの時の平均活動に変換

image_data_azimuth = image_data(:, is_align_azimuth(2,:));

damy = zeros(size(image_data_azimuth,1), max_stim_num);
repeat_num = size(image_data_azimuth,2)/max_stim_num;
for i = 1:repeat_num
    damy = damy + image_data_azimuth(:,max_stim_num*(i-1)+1:max_stim_num*i);
end
av_image_data_azimuth = damy./repeat_num;

image_data_altitude = image_data(:, is_align_altitude(2,:));

damy = zeros(size(image_data_altitude,1), max_stim_num);
repeat_num = size(image_data_altitude,2)/max_stim_num;
for i = 1:repeat_num
    damy = damy + image_data_altitude(:,max_stim_num*(i-1)+1:max_stim_num*i);
end
av_image_data_altitude = damy./repeat_num;

%%
figure;
plot(1:900, av_image_data_azimuth(:,:))
figure;
plot(1:900, av_image_data_altitude(:,:))

%% 視覚野の同定 平均でやっても全体でやっても同じ
% 4つの刺激方向のそれぞれについて、刺激トリガーによる平均△Fムービーを作成し、
% 各方向で10～40回の試行を平均した。方位と高度の位置マップを作成するため、
% 各ピクセルの蛍光対時間データから、フーリエ系列の第一高調波成分の位相から網膜視床位置を抽出した。

L = size(av_image_data_azimuth,2); % Length of signal
t = (0:L-1)*dt; % Time vector
ff = Fs*(0:(L/2))/L;
% フーリエ変換を実行
fourier_azimuth = fft(av_image_data_azimuth');
fourier_altitude = fft(av_image_data_altitude');

% 第1高調波成分を抽出
first_harmonic_idx = 2; % 1sで14度を使う
first_harmonic_azimuth = fourier_azimuth(first_harmonic_idx,:);
first_harmonic_altitude = fourier_altitude(first_harmonic_idx,:);
% 第1高調波成分の位相を抽出
phase_angle_azimuth = angle(first_harmonic_azimuth);
phase_angle_azimuth = reshape(phase_angle_azimuth,size(data,1),size(data,2));
phase_angle_altitude = angle(first_harmonic_altitude);
phase_angle_altitude = reshape(phase_angle_altitude,size(data,1),size(data,2));

[X,cmap] = imread("C:\Users\funam\Desktop\analysis\imaging\230808_Blue_tiff\230808_Blue_X1.tif");
% figure;
% imshow(X);

X = X(55:230,125:305);
X = ind2rgb(X,cmap);

figure;
imshow(X);
hold on;
contour(phase_angle_azimuth,'LevelStep', 0.2)
% contourf(phase_angle,'LevelStep', 0.1)
axis equal tight
colorbar
hold off;

figure;
imshow(X);
hold on;
contour(phase_angle_altitude,'LevelStep', 0.2)
% contourf(phase_angle,'LevelStep', 0.1)
axis equal tight
colorbar
hold off;

%% 
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
azimuth_center = sorted_indices(end);
azi_largest_indices = intersect(azi_largest_indices,selected_index);
% max of altitude
alt_diff = mean(av_image_data_altitude(:,zero_point+31:zero_point+91),2)...
            - mean(av_image_data_altitude(:,zero_point-59:zero_point),2);
[sorted_array, sorted_indices] = sort(alt_diff);
alt_largest_indices = sorted_indices(end-9999:end);
altitude_center = sorted_indices(end-100);
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

% figure;
% plot(1:900, av_image_data_azimuth(azimuth_center,:))
% figure;
% plot(1:900, av_image_data_altitude(altitude_center,:))

figure;
plot(1:900, av_image_data_azimuth(center,:))
figure;
plot(1:900, av_image_data_altitude(center,:))

%% altitude, azimuthごとに相互相関を計算

% 相互相関と位相ズレの計算
cross_corr_azimuth = zeros(size(av_image_data_azimuth,1),1);
for i = 1:size(av_image_data_azimuth,1)
    [cross_corr, lag] = xcorr(av_image_data_azimuth(center,:), av_image_data_azimuth(i,:)); % 相互相関
    [max_corr, max_idx] = max(cross_corr);   % 最大相関とそのインデックス
    phase_shift_samples = lag(max_idx);      % 位相ズレ（サンプル数）
    cross_corr_azimuth(i) = phase_shift_samples;
end

cross_corr_altitude = zeros(size(av_image_data_altitude,1),1);
for i = 1:size(av_image_data_altitude,1)
    [cross_corr, lag] = xcorr(av_image_data_altitude(center,:), av_image_data_altitude(i,:)); % 相互相関
    [max_corr, max_idx] = max(cross_corr);   % 最大相関とそのインデックス
    phase_shift_samples = lag(max_idx);      % 位相ズレ（サンプル数）
    cross_corr_altitude(i) = phase_shift_samples;
end

%%
% 結果の表示
cross_corr_azimuth = reshape(cross_corr_azimuth,size(data,1),size(data,2));
cross_corr_altitude = reshape(cross_corr_altitude,size(data,1),size(data,2));

figure;
imshow(X);
hold on;
contour(cross_corr_azimuth/100,'LevelStep', 0.1)
% contourf(phase_angle,'LevelStep', 0.1)
axis equal tight
colorbar
hold off;

figure;
imshow(X);
hold on;
contour(cross_corr_altitude/100,'LevelStep', 0.1)
% contourf(phase_angle,'LevelStep', 0.1)
axis equal tight
colorbar
hold off;

%% rest_dataをピクセル*時間に変換
rest_data_new = (rest_data-f)./f;
rest_image_data = zeros(size(rest_data,1)*size(rest_data,2),size(rest_data,3));
for i = 1:size(rest_data,3)
    rest_image_data(:,i) = reshape(rest_data_new(:,:,i),size(rest_data,1)*size(rest_data,2),1);
end

% L = size(rest_image_data,2); % Length of signal
% t = (0:L-1)*dt; % Time vector
% ff = Fs*(0:(L/2))/L;
% % フーリエ変換を実行
% fourier_rest = fft(rest_image_data');
% 
% % 第1高調波成分を抽出
% first_harmonic_idx = 2; % 1sで14度を使う
% first_harmonic_rest = fourier_rest(first_harmonic_idx,:);
% % 第1高調波成分の位相を抽出
% phase_angle_rest = angle(first_harmonic_rest);
% phase_angle_rest = reshape(phase_angle_rest,size(rest_data,1),size(rest_data,2));
% 
% [X,cmap] = imread("C:\Users\funam\Desktop\analysis\imaging\230806_Blue_tiff\230806_Blue_X1.tif");
% 
% X = X(55:230,125:305);
% X = ind2rgb(X,cmap);
% 
% figure;
% imshow(X);
% hold on;
% contour(phase_angle_rest,'LevelStep', 0.1)
% axis equal tight
% colorbar
% hold off;

% 相互相関と位相ズレの計算 どこを基準にとるかで大きく変わる
cross_corr_rest = zeros(size(rest_image_data,1),1);
for i = 1:size(rest_image_data,1)
    [cross_corr, lag] = xcorr(rest_image_data(center,:), rest_image_data(i,:)); % 相互相関
    [max_corr, max_idx] = max(cross_corr);   % 最大相関とそのインデックス
    phase_shift_samples = lag(max_idx);      % 位相ズレ（サンプル数）
    cross_corr_rest(i) = phase_shift_samples;
end

% 結果の表示
cross_corr_rest = reshape(cross_corr_rest,size(rest_data,1),size(rest_data,2));

%% scatterでかっこよく表示

% 配列のサイズを取得
[num_rows, num_cols] = size(cross_corr_rest);

% x座標とy座標のグリッドを生成
[a, b] = meshgrid(1:num_cols, 1:num_rows);

figure;
imshow(X);
hold on;
contour(cross_corr_rest)
% contourf(cross_corr_rest,'LevelStep', 0.1)
% scatter(a(:), b(:), 2, cross_corr_rest(:), 'filled');
axis equal tight
colorbar
hold off;

%%

% % 仮想的な時系列データを生成
% t = 1:100; % 時間軸
% noise1 = 0.2*randn(size(t));
% noise2 = -0.5*randn(size(t));
% data1 = [sin(2*t) + noise1; sin(2*t) - noise1]; % 1つ目の時系列データ
% data2 = [2*sin(2*t-0.04) + noise2; 2*sin(2*t-0.04) - noise2]; % 2つ目の時系列データ
% 
% % カノニカル相関分析を実行
% [Coefficients, Score, Latent] = canoncorr(data1', data2');
% 
% % 結果の表示
% disp('カノニカル相関係数:');
% disp(Coefficients);
% disp('カノニカル変数のスコア:');
% disp(Score);
% disp('ラテント変数の分散:');
% disp(Latent);

%% カノニカル相関 上の実験より位相差も検出してくれる 正負のズレに対して同様ではない 周期に対して位相ズレが小さければほぼイコールになる？
% altitude
image_data_altitude_devide = zeros(size(image_data_altitude,1),max_stim_num,repeat_num);
for i = 1:repeat_num
    image_data_altitude_devide(:,:,i) = image_data_altitude(:,max_stim_num*(i-1)+1:max_stim_num*i);
end
image_data_altitude_devide = permute(image_data_altitude_devide,[3,2,1]);

canon_corr_altitude = zeros(size(image_data_altitude_devide,3),1);
for i = 1:size(image_data_altitude_devide,3)
    [a,b,c] = canoncorr(image_data_altitude_devide(:,:,center)', image_data_altitude_devide(:,:,i)');
    canon_corr_altitude(i) = c(1);
end

canon_corr_altitude = reshape(canon_corr_altitude,size(rest_data,1),size(rest_data,2));

% azimuth
image_data_azimuth_devide = zeros(size(image_data_azimuth,1),max_stim_num,repeat_num);
for i = 1:repeat_num
    image_data_azimuth_devide(:,:,i) = image_data_azimuth(:,max_stim_num*(i-1)+1:max_stim_num*i);
end
image_data_azimuth_devide = permute(image_data_azimuth_devide,[3,2,1]);

canon_corr_azimuth = zeros(size(image_data_azimuth_devide,3),1);
for i = 1:size(image_data_azimuth_devide,3)
    [a,b,c] = canoncorr(image_data_azimuth_devide(:,:,center)', image_data_azimuth_devide(:,:,i)');
    canon_corr_azimuth(i) = c(1);
end

canon_corr_azimuth = reshape(canon_corr_azimuth,size(rest_data,1),size(rest_data,2));

%%
figure;
imshow(X);
hold on;
contour(canon_corr_altitude,'LevelStep', 0.005)
axis equal tight
colorbar
clim([0.9 1])
scatter(centroids(1),centroids(2))
hold off;

figure;
imshow(X);
hold on;
contour(canon_corr_azimuth,'LevelStep', 0.005)
axis equal tight
colorbar
clim([0.9 1])
hold off;


% 端の点を求めて、等高線を等間隔で引いて、いい感じにscatterで色付け











%% 60Hzは0.01666s=17msごとのサンプリングなので、刺激提示後6 frames (約100ms分)以上とってくる

ind_altitude = zeros(size(altitude_num,3),1);
for i = 1:size(altitude_num,3)
    k = myFind(altitude_judge_real(i,:),1,5,1); % 5フレームに変えた
    l = myFind(altitude_judge_real(i,:),0,20,1);
    for j = 1:size(l,1)-1
        temp = k > l(j) & k < l(j+1);
        ind_altitude(i,j) = min(k(temp));
    end
end

ind_azimuth = zeros(size(azimuth_num,3),1);
for i = 1:size(azimuth_num,3)
    k = myFind(azimuth_judge_real(i,:),1,5,1);
    l = myFind(azimuth_judge_real(i,:),0,20,1);
    for j = 1:size(l,1)-1
        temp = k > l(j) & k < l(j+1);
        ind_azimuth(i,j) = min(k(temp));
    end
end

%% 刺激のonsetの前後100 framesを取ってきて、特定のaltitude, azimuthに反応したピクセルを抽出する
num_frames = 200;

% azimuth
all_data_1 = zeros(size(image_data,1),num_frames,size(ind_azimuth,2),size(azimuth_num,3));% [ピクセル数 フレーム数 イベント数 フィールド数]
av_data_1 = zeros(size(image_data,1),num_frames,size(azimuth_num,3));
std_data_1 = zeros(size(image_data,1),num_frames,size(azimuth_num,3));

% 端には刺激が出ていない
for i = 3:14
    for j = 1:num_frames
        temp = image_data(:,is_align_azimuth(2,ind_azimuth(i,:))+j-num_frames/2); % 230724 ここ変えた
        all_data_1(:,j,:,i) = temp;
        av_data_1(:,j,i) = mean(temp,2);
        std_data_1(:,j,i) = std(temp,0,2);
    end
end

% altitude
all_data_2 = zeros(size(image_data,1),num_frames,size(ind_altitude,2),size(altitude_num,3));% [ピクセル数 フレーム数 イベント数 フィールド数]
av_data_2 = zeros(size(image_data,1),num_frames,size(altitude_num,3));
std_data_2 = zeros(size(image_data,1),num_frames,size(altitude_num,3));

% 端には刺激が出ていない
for i = 4:15
    for j = 1:num_frames
        temp = image_data(:,is_align_altitude(2,ind_altitude(i,:))+j-num_frames/2); % 230724 ここ変えた
        all_data_2(:,j,:,i) = temp;
        av_data_2(:,j,i) = mean(temp,2);
        std_data_2(:,j,i) = std(temp,0,2);
    end
end

%%
% onset 59 frame 前からとonset 60 frame後までで変化があったか

% azimuth
all_data_devide_1 = permute(all_data_1,[3 2 1 4]);% [イベント数 フレーム数 ピクセル数 フィールド数]
av_data_devide_1 = zeros(size(all_data_1,3),2,size(all_data_1,1),size(azimuth_num,1));

% 端には刺激が出ていない
for i = 3:14
    for j = 1:size(all_data_devide_1,3)
        av_data_devide_1(:,:,j,i) = [mean(all_data_devide_1(:,1:num_frames/2,j,i),2), mean(all_data_devide_1(:,num_frames/2+1:num_frames,j,i),2)];
    end
end

% altitude
all_data_devide_2 = permute(all_data_2,[3 2 1 4]);% [イベント数 フレーム数 ピクセル数 フィールド数]
av_data_devide_2 = zeros(size(all_data_2,3),2,size(all_data_2,1),size(altitude_num,1));

% 端には刺激が出ていない
for i = 4:15
    for j = 1:size(all_data_devide_2,3)
        av_data_devide_2(:,:,j,i) = [mean(all_data_devide_2(:,1:num_frames/2,j,i),2), mean(all_data_devide_2(:,num_frames/2+1:num_frames,j,i),2)];
    end
end

%%

% azimuth
responce_pixel_num_1 = cell(1,size(azimuth_num,3));
% 端には刺激が出ていない
for i = 3:14
    temp = [];
    for j = 1:size(av_data_devide_1,3)
        [p,h] = signrank(av_data_devide_1(:,1,j,i), av_data_devide_1(:,2,j,i),"alpha",0.01);
        if h==1
            temp(1,end+1) = j;
            temp(2,end) = p;
        end
    end
    responce_pixel_num_1(i) = {temp};
end

% altitude
responce_pixel_num_2 = cell(1,size(altitude_num,3));
% 端には刺激が出ていない
for i = 4:15
    temp = [];
    for j = 1:size(av_data_devide_2,3)
        [p,h] = signrank(av_data_devide_2(:,1,j,i), av_data_devide_2(:,2,j,i),"alpha",0.01);
        if h==1
            temp(1,end+1) = j;
            temp(2,end) = p;
        end
    end
    responce_pixel_num_2(i) = {temp};
end

%% 検定をせずに上位100ピクセルをとる場合

% azimuth
res_pixels_1 = zeros(size(data,1),size(data,2),size(azimuth_num,3));

% 端には刺激が出ていない
for i = 3:14
    array = permute(mean(av_data_devide_1(:,2,:,i) - av_data_devide_1(:,1,:,i), 1), [3 2 1]);
    [sorted_array, sorted_indices] = sort(array);
    largest_indices = sorted_indices(end-99:end);
%     smallest_indices = sorted_indices(1:100);
    temp = zeros(size(av_data_devide_1,3),1);
    temp(largest_indices) = 1;
%     temp(smallest_indices) = 1;
    temp = reshape(temp,size(data,1),size(data,2));
    res_pixels_1(:,:,i) = temp;
end

% res_R_1 = X(:,:,1);
% res_G_1 = X(:,:,2);
% res_B_1 = X(:,:,3);
% % 端には刺激が出ていない
% for i = 3:14
%     temp = res_pixels_1(:,:,i);
%     res_R_1(temp==1) = colour_1(i,1);
%     res_G_1(temp==1) = colour_1(i,2);
%     res_B_1(temp==1) = colour_1(i,3);
% end
% figure;
% imshow(cat(3,res_R_1,res_G_1,res_B_1))

% altitude
res_pixels_2 = zeros(size(data,1),size(data,2),size(altitude_num,3));

% 端には刺激が出ていない
for i = 4:15
    array = permute(mean(av_data_devide_2(:,2,:,i) - av_data_devide_2(:,1,:,i), 1), [3 2 1]);
    [sorted_array, sorted_indices] = sort(array);
    largest_indices = sorted_indices(end-99:end);
%     smallest_indices = sorted_indices(1:100);
    temp = zeros(size(av_data_devide_2,3),1);
    temp(largest_indices) = 1;
%     temp(smallest_indices) = 1;
    temp = reshape(temp,size(data,1),size(data,2));
    res_pixels_2(:,:,i) = temp;
end

% res_R_2 = X(:,:,1);
% res_G_2 = X(:,:,2);
% res_B_2 = X(:,:,3);
% % 端には刺激が出ていない
% for i = 4:15
%     temp = res_pixels_2(:,:,i);
%     res_R_2(temp==1) = colour_2(i,1);
%     res_G_2(temp==1) = colour_2(i,2);
%     res_B_2(temp==1) = colour_2(i,3);
% end
% figure;
% imshow(cat(3,res_R_2,res_G_2,res_B_2))

%% 統計的に有意かつ変化も大きい

% azimuth
res_R_1 = X(:,:,1);
res_G_1 = X(:,:,2);
res_B_1 = X(:,:,3);

% 端には刺激が出ていない
for i = 3:14
    array = cell2mat(responce_pixel_num_1(i));
    if isempty(array)==0
        temp = zeros(size(av_data_devide_1,3),1);
        temp(array(1,:)) = 1;
        temp = reshape(temp,size(data,1),size(data,2))+res_pixels_1(:,:,i);
        res_R_1(temp==2) = colour_1(i,1);
        res_G_1(temp==2) = colour_1(i,2);
        res_B_1(temp==2) = colour_1(i,3);
    end
end
figure;
imshow(cat(3,res_R_1,res_G_1,res_B_1))

% altitude
res_R_2 = X(:,:,1);
res_G_2 = X(:,:,2);
res_B_2 = X(:,:,3);

% 端には刺激が出ていない
for i = 4:15
    array = cell2mat(responce_pixel_num_2(i));
    if isempty(array)==0
        temp = zeros(size(av_data_devide_2,3),1);
        temp(array(1,:)) = 1;
        temp = reshape(temp,size(data,1),size(data,2))+res_pixels_2(:,:,i);
        res_R_2(temp==2) = colour_2(i,1);
        res_G_2(temp==2) = colour_2(i,2);
        res_B_2(temp==2) = colour_2(i,3);
    end
end
figure;
imshow(cat(3,res_R_2,res_G_2,res_B_2))

%% 統計的に有意かつ変化も大きい

% azimuth
res_R_1 = X(:,:,1);
res_G_1 = X(:,:,2);
res_B_1 = X(:,:,3);

% 端には刺激が出ていない
% 3から順に刺激を出しているので14の方から解析
for i = 3:14
    array = cell2mat(responce_pixel_num_1(17-i));
    if isempty(array)==0
        temp = zeros(size(av_data_devide_1,3),1);
        temp(array(1,:)) = 1;
        temp = reshape(temp,size(data,1),size(data,2))+res_pixels_1(:,:,17-i);
        ttemp = zeros(size(data,1),size(data,2));
        ttemp(temp==2) = 1;
        [centroids, shrunken_matrix] = shrink_regions_around_centroids(ttemp, 100);
        res_R_1(shrunken_matrix==1) = colour_1(17-i,1);
        res_G_1(shrunken_matrix==1) = colour_1(17-i,2);
        res_B_1(shrunken_matrix==1) = colour_1(17-i,3);
    end
end

figure;
imshow(cat(3,res_R_1,res_G_1,res_B_1))
hold on;
% contour(phase_angle_1,'LevelStep', 0.1)
% contourf(phase_angle,'LevelStep', 0.1)
% axis equal tight
% colorbar
hold off;

% altitude
res_R_2 = X(:,:,1);
res_G_2 = X(:,:,2);
res_B_2 = X(:,:,3);

% 端には刺激が出ていない
% 15から順に刺激を出しているので4の方から解析
for i = 4:15
    array = cell2mat(responce_pixel_num_2(i));
    if isempty(array)==0
        temp = zeros(size(av_data_devide_2,3),1);
        temp(array(1,:)) = 1;
        temp = reshape(temp,size(data,1),size(data,2))+res_pixels_2(:,:,i);
        ttemp = zeros(size(data,1),size(data,2));
        ttemp(temp==2) = 1;
        [centroids, shrunken_matrix] = shrink_regions_around_centroids(ttemp, 100);
        res_R_2(shrunken_matrix==1) = colour_2(i,1);
        res_G_2(shrunken_matrix==1) = colour_2(i,2);
        res_B_2(shrunken_matrix==1) = colour_2(i,3);
    end
end
figure;
imshow(cat(3,res_R_2,res_G_2,res_B_2))
hold on;
% contour(phase_angle_2,'LevelStep', 0.1)
% contourf(phase_angle,'LevelStep', 0.1)
% axis equal tight
% colorbar
hold off;

%% altitudeとazimuthを使用してgridにする

% res_R = X(:,:,1);
% res_G = X(:,:,2);
% res_B = X(:,:,3);
% 
% % 1,9には刺激が出ていない
% for i = 4:6
%     for j = 4:6
%         array_1 = cell2mat(responce_pixel_num_1(i)); array_2 = cell2mat(responce_pixel_num_2(j));
%         temp_1 = zeros(size(av_data_devide_1,3),1); temp_2 = zeros(size(av_data_devide_2,3),1);
%         temp_1(array_1(1,:)) = 1; temp_2(array_2(1,:)) = 1;
%         temp_1 = reshape(temp_1,size(data,1),size(data,2))+res_pixels_1(:,:,i);
%         temp_2 = reshape(temp_2,size(data,1),size(data,2))+res_pixels_2(:,:,j);
%         temp = temp_1+temp_2;
%         res_R(temp==4) = colour(i-3+3*(j-4),1);
%         res_G(temp==4) = colour(i-3+3*(j-4),2);
%         res_B(temp==4) = colour(i-3+3*(j-4),3);
%     end
% end
% figure;
% imshow(cat(3,res_R,res_G,res_B))

%% 検定のみ

% azimuth
res_R_1 = X(:,:,1);
res_G_1 = X(:,:,2);
res_B_1 = X(:,:,3);

% 端には刺激が出ていない
% 3から順に刺激を出しているので14の方から解析
for i = 3:14
    array = cell2mat(responce_pixel_num_1(17-i));
    if isempty(array)==0
        temp = zeros(size(av_data_devide_1,3),1);
        temp(array(1,:)) = 1;
        temp = reshape(temp,size(data,1),size(data,2));
        ttemp = zeros(size(data,1),size(data,2));
        ttemp(temp==1) = 1;
        [centroids, shrunken_matrix] = shrink_regions_around_centroids(ttemp, 100);
        res_R_1(shrunken_matrix==1) = colour_1(17-i,1);
        res_G_1(shrunken_matrix==1) = colour_1(17-i,2);
        res_B_1(shrunken_matrix==1) = colour_1(17-i,3);
    end
end

figure;
imshow(cat(3,res_R_1,res_G_1,res_B_1))
hold on;
% contour(phase_angle_1,'LevelStep', 0.1)
% contourf(phase_angle,'LevelStep', 0.1)
% axis equal tight
% colorbar
hold off;

% altitude
res_R_2 = X(:,:,1);
res_G_2 = X(:,:,2);
res_B_2 = X(:,:,3);

% 端には刺激が出ていない
% 15から順に刺激を出しているので4の方から解析
for i = 4:15
    array = cell2mat(responce_pixel_num_2(i));
    if isempty(array)==0
        temp = zeros(size(av_data_devide_2,3),1);
        temp(array(1,:)) = 1;
        temp = reshape(temp,size(data,1),size(data,2));
        ttemp = zeros(size(data,1),size(data,2));
        ttemp(temp==1) = 1;
        [centroids, shrunken_matrix] = shrink_regions_around_centroids(ttemp, 100);
        res_R_2(shrunken_matrix==1) = colour_2(i,1);
        res_G_2(shrunken_matrix==1) = colour_2(i,2);
        res_B_2(shrunken_matrix==1) = colour_2(i,3);
    end
end
figure;
imshow(cat(3,res_R_2,res_G_2,res_B_2))
hold on;
% contour(phase_angle_2,'LevelStep', 0.1)
% contourf(phase_angle,'LevelStep', 0.1)
% axis equal tight
% colorbar
hold off;

%%
% 6Hzのみを使う, →1sで14度を使う
fourier_1_select = zeros(size(fourier_azimuth,1), size(fourier_azimuth,2));
fourier_1_select(86:96,:) = fourier_azimuth(86:96,:);
fourier_2_select = zeros(size(fourier_altitude,1), size(fourier_altitude,2));
fourier_2_select(86:96,:) = fourier_altitude(86:96,:);

ifourier_1 = fft(fourier_1_select);
ifourier_2 = fft(fourier_2_select);

ifourier_1 = abs(ifourier_1');
ifourier_2 = abs(ifourier_2');

figure;
plot(1:900, av_image_data_azimuth(1000,:))
figure;
plot(1:900, ifourier_1(1000,:))

%%
num_frames = 100;

% azimuth
% その領域に刺激が出る最初のindexを取ってくる
% 1s後、60frame後との差が大きい100ピクセルとってくる

azimuth_6Hz = zeros(size(ifourier_1,1),num_frames,size(azimuth_judge,1));

% 端には刺激が出ていない
for i = 3:14
    I = find(azimuth_judge(i,:)==1);
    I = I(1);
    for j = 1:num_frames
        azimuth_6Hz(:,j,i) = ifourier_1(:,I+j-num_frames/2);
    end
end

% altitude
altitude_6Hz = zeros(size(ifourier_2,1),num_frames,size(altitude_judge,1));

% 端には刺激が出ていない
for i = 4:15
    I = find(altitude_judge(i,:)==1);
    I = I(1);
    for j = 1:num_frames
        altitude_6Hz(:,j,i) = ifourier_2(:,I+j-num_frames/2);
    end
end

av_azimuth_6Hz = zeros(size(azimuth_6Hz,1),2,size(azimuth_judge,1));
% 端には刺激が出ていない
for i = 3:14
    av_azimuth_6Hz(:,:,i) = [mean(azimuth_6Hz(:,1:num_frames/2,i),2), mean(azimuth_6Hz(:,num_frames/2+1:num_frames,i),2)];
end

% altitude
av_altitude_6Hz = zeros(size(altitude_6Hz,1),2,size(altitude_judge,1));

% 端には刺激が出ていない
for i = 4:15
    av_altitude_6Hz(:,:,i) = [mean(altitude_6Hz(:,1:num_frames/2,i),2), mean(altitude_6Hz(:,num_frames/2+1:num_frames,i),2)];
end

% azimuth
responce_azimuth_6Hz = zeros(size(data,1),size(data,2),size(azimuth_judge,1));

% 端には刺激が出ていない
for i = 3:14
    array = av_azimuth_6Hz(:,2,i)-av_azimuth_6Hz(:,1,i);
    [sorted_array, sorted_indices] = sort(array);
    largest_indices = sorted_indices(end-399:end);
    temp = zeros(size(av_azimuth_6Hz,1),1);
    temp(largest_indices) = 1;
    temp = reshape(temp,size(data,1),size(data,2));
    responce_azimuth_6Hz(:,:,i) = temp;
end

% altitude
responce_altitude_6Hz = zeros(size(data,1),size(data,2),size(altitude_judge,1));

% 端には刺激が出ていない
for i = 4:15
    array = av_altitude_6Hz(:,2,i)-av_altitude_6Hz(:,1,i);
    [sorted_array, sorted_indices] = sort(array);
    largest_indices = sorted_indices(end-399:end);
    temp = zeros(size(av_altitude_6Hz,1),1);
    temp(largest_indices) = 1;
    temp = reshape(temp,size(data,1),size(data,2));
    responce_altitude_6Hz(:,:,i) = temp;
end

%%
res_R_1 = X(:,:,1);
res_G_1 = X(:,:,2);
res_B_1 = X(:,:,3);
% 端には刺激が出ていない
% 3から順に刺激を出しているので14の方から解析
for i = 3:14
    temp = responce_azimuth_6Hz(:,:,17-i);
    [centroids, shrunken_matrix] = shrink_regions_around_centroids(temp,100);
    res_R_1(shrunken_matrix==1) = colour_1(17-i,1);
    res_G_1(shrunken_matrix==1) = colour_1(17-i,2);
    res_B_1(shrunken_matrix==1) = colour_1(17-i,3);
end
figure;
imshow(cat(3,res_R_1,res_G_1,res_B_1))

res_R_2 = X(:,:,1);
res_G_2 = X(:,:,2);
res_B_2 = X(:,:,3);
% 端には刺激が出ていない
% 15から順に刺激を出しているので4の方から解析
for i = 4:15
    temp = responce_altitude_6Hz(:,:,i);
    [centroids, shrunken_matrix] = shrink_regions_around_centroids(temp,100);
    res_R_2(shrunken_matrix==1) = colour_2(i,1);
    res_G_2(shrunken_matrix==1) = colour_2(i,2);
    res_B_2(shrunken_matrix==1) = colour_2(i,3);
end
figure;
imshow(cat(3,res_R_2,res_G_2,res_B_2))

%%
function num = myFind(T,threshold,n,m)
% threshold：閾値
% n：何行以上続くか
% m：何番目の番号を取得するか
x = T' == threshold;
a = cell2mat(arrayfun(@(t)1:t,diff(find([1 diff(x(:,1)') 1])),'un',0))';
num = find(a==n & x==1) - (n-m);
num = num(x(num));
end

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

















