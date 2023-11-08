% imds = imageDatastore("C:\Users\funam\Desktop\analysis\imaging\230722_Blue_altitude_2_tiff");
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
%     raw_data(:,:,i) = imresize(temp, 1/4);
%     %raw_data(:,:,i) = temp;
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
% 
% %%
% save_filename = "230722_Blue_altitude_2_4";
% save(save_filename+".mat",'raw_data','iteNum',"-v7.3");

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

altitude_num = zeros(size(y,1), size(y,2), 9);
altitude_num_RGB = zeros(size(y,1), size(y,2), 3, 9);
colour = zeros(9,3);
colour(1,:) = [1 0 0];
colour(2,:) = [1 0.5 0];
colour(3,:) = [1 1 0];
colour(4,:) = [0.5 1 0];
colour(5,:) = [0 1 0];
colour(6,:) = [0 1 0.5];
colour(7,:) = [0 1 1];
colour(8,:) = [0 0.5 1];
colour(9,:) = [0 0 1];

for i = 1:9
    ind = phi >= (-90+20*(i-1))/180*pi & phi < (-90+20*i)/180*pi;
    temp = zeros(size(y,1), size(y,2));
    temp_R = temp; temp_G = temp; temp_B = temp;
    temp(ind) = 1;
    temp_R(ind) = colour(i,1);
    temp_G(ind) = colour(i,2);
    temp_B(ind) = colour(i,3);
    altitude_num(:,:,i) = temp;
    altitude_num_RGB(:,:,:,i) = cat(3,temp_R,temp_G,temp_B);
    subplot(3,3,i);
    imshow(temp)
end

figure;
imshow(sum(altitude_num_RGB,4));

%% azimuthにnumbering
% G,R,Bの順に動かす

azimuth_num = zeros(size(y,1), size(y,2), 9);
azimuth_num_RGB = zeros(size(y,1), size(y,2), 3, 9);

for i = 1:9
    ind = theta >= (90-20*i)/180*pi & theta < (90-20*(i-1))/180*pi;
    temp = zeros(size(y,1), size(y,2));
    temp_R = temp; temp_G = temp; temp_B = temp;
    temp(ind) = 1;
    temp_R(ind) = colour(i,1);
    temp_G(ind) = colour(i,2);
    temp_B(ind) = colour(i,3);
    azimuth_num(:,:,i) = temp;
    azimuth_num_RGB(:,:,:,i) = cat(3,temp_R,temp_G,temp_B);
    subplot(3,3,i);
    imshow(temp)
end

figure;
imshow(sum(azimuth_num_RGB,4));

%% fieldにnumbering
% G,R,Bの順に動かす

field_num = zeros(size(y,1), size(y,2), 9);
field_num_RGB = zeros(size(y,1), size(y,2), 3, 9);

for i = -1:1
    for j = -1:1
        ind = theta >= (-10-i*20)/180*pi & theta < (-10+(-i+1)*20)/180*pi...
            & phi >= (-10+j*20)/180*pi & phi < (-10+(j+1)*20)/180*pi;
        temp = zeros(size(y,1), size(y,2));
        temp_R = temp; temp_G = temp; temp_B = temp;
        temp(ind) = 1;
        temp_R(ind) = colour((i+2)*3-1+j,1);
        temp_G(ind) = colour((i+2)*3-1+j,2);
        temp_B(ind) = colour((i+2)*3-1+j,3);
        field_num(:,:,(i+2)*3-1+j) = temp;
        field_num_RGB(:,:,:,(i+2)*3-1+j) = cat(3,temp_R,temp_G,temp_B);
        subplot(3,3,(i+2)*3-1+j);
        imshow(temp)
    end
end

figure;
imshow(sum(field_num_RGB,4));

%% 3を超えるところと0.25を超えるところをresting stateとした
load('230711_Blue_3_4.mat')
load('NI_11-Jul-2023_2237_11.bin.mat');
rest_duration = find(ch(7,:,:)>0.2, 1 ) - find(ch(1,:,:)>4, 1 );
rest_data = raw_data(:,:,1:rest_duration);
f = mean(rest_data,3);

%% df/fを計算
data = (raw_data-f)./f;

%%

Fs = 60; % sampling frequency in Hz
dt = 1/Fs; % time step in seconds
T = length(raw_data); % number of time points
t = (0:T-1)*dt; % time vector
% 
% figure
% plot(t, yy)

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
estim_num_stimuli = [];
estim_num_stimuli(1) = num_stimuli(1);
for i = 1:length(num_stimuli)-1
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
end

%%

% カメラのタイミングとイメージング画像のindex set
num_images(2,:) = 1:length(num_images);

% 刺激のタイミングと提示刺激のindex set
max_stim_num = 1800;
estim_num_stimuli(2,:) = mod(0:(length(estim_num_stimuli)-1), max_stim_num) + 1;

% イメージング画像のindex と提示刺激のindexをalign
align_point = knnsearch(num_images(1,:)', estim_num_stimuli(1,1));
align_length = min(length(estim_num_stimuli), length(num_images)-align_point+1);

image_stim_align(1,:) = estim_num_stimuli(2,1:align_length);
image_stim_align(2,:) = num_images(2,align_point:align_point+align_length-1);

%% 1-9のaltitude, azimuthに刺激が出ていたかどうか

altitude_judge = zeros(size(altitude_num,3),max_stim_num);
azimuth_judge = zeros(size(azimuth_num,3),max_stim_num);
for i = 1:max_stim_num
    if i > max_stim_num/2
        temp = load("stim_"+num2str(i));
        temp = temp.n;
        temp(1:50, end-50:end) = 0;
        for ii = 1:size(altitude_num,3)
            ttemp = temp+altitude_num(:,:,ii);
            altitude_judge(ii,i) = abs(isempty(find(ttemp==2, 1))-1);
        end
    else
        temp = load("stim_"+num2str(i));
        temp = temp.n;
        temp(1:50, end-50:end) = 0;
        for ii = 1:size(azimuth_num,3)
            ttemp = temp+azimuth_num(:,:,ii);
            azimuth_judge(ii,i) = abs(isempty(find(ttemp==2, 1))-1);
        end
    end
end

%% 実際に提示した刺激の1-9のaltitude, azimuthに刺激があったかどうか

altitude_judge_real = altitude_judge;
azimuth_judge_real = azimuth_judge;
r = rem(length(image_stim_align), max_stim_num);
q = (length(image_stim_align)-r)/max_stim_num;

for i = 1:q-1
    altitude_judge_real = cat(2,altitude_judge_real,altitude_judge);
    azimuth_judge_real = cat(2,azimuth_judge_real,azimuth_judge);
end
altitude_judge_real = cat(2,altitude_judge_real,altitude_judge(:,1:r));
azimuth_judge_real = cat(2,azimuth_judge_real,azimuth_judge(:,1:r));

%%
image_data = zeros(size(data,1)*size(data,2),size(data,3));
for i = 1:size(data,3)
    image_data(:,i) = reshape(data(:,:,i),size(data,1)*size(data,2),1);
end

% figure;
% plot(t, image_data)

%%

av_image_data = image_data(:,image_stim_align(2,1):image_stim_align(2,end));
av_image_data = av_image_data(:,1:size(av_image_data,2)-rem(size(av_image_data,2), max_stim_num));

damy = zeros(size(av_image_data,1), max_stim_num);
repeat_num = size(av_image_data,2)/max_stim_num;
for i = 1:repeat_num
    damy = damy + av_image_data(:,max_stim_num*(i-1)+1:max_stim_num*i);
end
av_image_data = damy./repeat_num;

%%
% figure;
% plot(1:1800, av_image_data(1411,1:1800))

figure;
plot(1:1800, movmean(av_image_data(2416,1:1800), 10, 'Endpoints', 'shrink'))

%% 60Hzは0.01666s=17msごとのサンプリングなので、刺激提示後6 frames (約100ms分)以上とってくる

ind_altitude = zeros(9,1);
for i = 1:9
    % altitudeに刺激が10 frame以上でているところの最初のindexを取ってくる
    k = myFind(altitude_judge_real(i,:),1,10,1);
    l = myFind(altitude_judge_real(i,:),0,20,1);
    for j = 1:size(l,1)-1
        temp = k > l(j) & k < l(j+1);
        ind_altitude(i,j) = min(k(temp));
    end
end

ind_azimuth = zeros(9,1);
for i = 1:9
    % azimuthに刺激が10 frame以上でているところの最初のindexを取ってくる
    k = myFind(azimuth_judge_real(i,:),1,10,1);
    l = myFind(azimuth_judge_real(i,:),0,20,1);
    for j = 1:size(l,1)-1
        temp = k > l(j) & k < l(j+1);
        ind_azimuth(i,j) = min(k(temp));
    end
end

%%
% num_frames = 15;
% time = (-5:num_frames-6)*dt;
% time2 = [time, fliplr(time)];
%num_frames = 200;
num_frames = 180;
%time = (-99:num_frames-100)*dt;
time = (-89:num_frames-90)*dt;
time2 = [time, fliplr(time)];

%% 刺激のonsetの前後100 framesを取ってきて、特定のaltitude, azimuthに反応したピクセルを抽出する
% 刺激のonsetの前後100 framesを取ってくる

% altitude
all_data_1 = zeros(size(image_data,1),num_frames,size(ind_altitude,2),9);% [ピクセル数 フレーム数 イベント数 フィールド数]
av_data = zeros(size(image_data,1),num_frames,9);
std_data = zeros(size(image_data,1),num_frames,9);

% 1,9には刺激が出ていない
for i = 2:8
    for j = 1:num_frames
        %temp = image_data(:,ind_altitude(i,:)+j-100);
        temp = image_data(:,ind_altitude(i,:)+j-90);
        all_data_1(:,j,:,i) = temp;
        av_data(:,j,i) = mean(temp,2);
        std_data(:,j,i) = std(temp,0,2);
    end
end

% azimuth
all_data_2 = zeros(size(image_data,1),num_frames,size(ind_azimuth,2),9);% [ピクセル数 フレーム数 イベント数 フィールド数]
av_data = zeros(size(image_data,1),num_frames,9);
std_data = zeros(size(image_data,1),num_frames,9);

% 1,9には刺激が出ていない
for i = 2:8
    for j = 1:num_frames
        %temp = image_data(:,ind_azimuth(i,:)+j-100);
        temp = image_data(:,ind_azimuth(i,:)+j-90);
        all_data_2(:,j,:,i) = temp;
        av_data(:,j,i) = mean(temp,2);
        std_data(:,j,i) = std(temp,0,2);
    end
end
%%
% onset 99 frame 前からとonset 100 frame後までで変化があったか

% altitude
all_data_devide_1 = permute(all_data_1,[3 2 1 4]);% [イベント数 フレーム数 ピクセル数 フィールド数]
av_data_devide_1 = zeros(size(all_data_1,3),2,size(all_data_1,1),9);

% 1,9には刺激が出ていない
for i = 2:8
    for j = 1:size(all_data_devide_1,3)
        %av_data_devide_1(:,:,j,i) = [mean(all_data_devide_1(:,1:100,j,i),2), mean(all_data_devide_1(:,101:num_frames,j,i),2)];
        av_data_devide_1(:,:,j,i) = [mean(all_data_devide_1(:,1:90,j,i),2), mean(all_data_devide_1(:,91:num_frames,j,i),2)];
    end
end

% azimuth
all_data_devide_2 = permute(all_data_2,[3 2 1 4]);% [イベント数 フレーム数 ピクセル数 フィールド数]
av_data_devide_2 = zeros(size(all_data_2,3),2,size(all_data_2,1),9);

% 1,9には刺激が出ていない
for i = 2:8
    for j = 1:size(all_data_devide_2,3)
        %av_data_devide_2(:,:,j,i) = [mean(all_data_devide_2(:,1:100,j,i),2), mean(all_data_devide_2(:,101:num_frames,j,i),2)];
        av_data_devide_2(:,:,j,i) = [mean(all_data_devide_2(:,1:90,j,i),2), mean(all_data_devide_2(:,91:num_frames,j,i),2)];
    end
end

%%

% altitude
responce_pixel_num_1 = cell(1,9);
% 1,9には刺激が出ていない
for i = 2:8
    temp = [];
    for j = 1:size(av_data_devide_1,3)
        [p,h] = signrank(av_data_devide_1(:,1,j,i), av_data_devide_1(:,2,j,i),'tail','left',"alpha",0.005);
%         [p,h] = signrank(av_data_devide_1(:,1,j,i), av_data_devide_1(:,2,j,i),"alpha",0.01);
        if h==1
            temp(1,end+1) = j;
            temp(2,end) = p;
        end
    end
    responce_pixel_num_1(i) = {temp};
end

% azimuth
responce_pixel_num_2 = cell(1,9);
% 1,9には刺激が出ていない
for i = 2:8
    temp = [];
    for j = 1:size(av_data_devide_2,3)
        [p,h] = signrank(av_data_devide_2(:,1,j,i), av_data_devide_2(:,2,j,i),'tail','left',"alpha",0.005);
%         [p,h] = signrank(av_data_devide_2(:,1,j,i), av_data_devide_2(:,2,j,i),"alpha",0.01);
        if h==1
            temp(1,end+1) = j;
            temp(2,end) = p;
        end
    end
    responce_pixel_num_2(i) = {temp};
end

%% 検定をせずに上位300ピクセルをとる場合

[X,cmap] = imread("C:\Users\funam\Desktop\analysis\imaging\230711_Blue_3_tiff\230711_Blue_3_tiff_X1.tif");
X = imresize(X, 1/4);
X = ind2rgb(X,cmap);
% imshow(X);

% altitude
res_pixels_1 = zeros(size(data,1),size(data,2),9);

% 1,9には刺激が出ていない
for i = 2:8
    array = permute(mean(av_data_devide_1(:,2,:,i) - av_data_devide_1(:,1,:,i), 1), [3 2 1]);
    [sorted_array, sorted_indices] = sort(array);
    largest_indices = sorted_indices(end-149:end);
    %smallest_indices = sorted_indices(1:150);
    temp = zeros(size(av_data_devide_1,3),1);
    temp(largest_indices) = 1;
    %temp(smallest_indices) = 1;
    temp = reshape(temp,size(data,1),size(data,2));
    res_pixels_1(:,:,i) = temp;
end

res_R = X(:,:,1);
res_G = X(:,:,2);
res_B = X(:,:,3);
% 1,9には刺激が出ていない
for i = 2:8
    temp = res_pixels_1(:,:,i);
    res_R(temp==1) = colour(i,1);
    res_G(temp==1) = colour(i,2);
    res_B(temp==1) = colour(i,3);
end
% figure;
% imshow(cat(3,res_R,res_G,res_B))

% azimuth
res_pixels_2 = zeros(size(data,1),size(data,2),9);

% 1,9には刺激が出ていない
for i = 2:8
    array = permute(mean(av_data_devide_2(:,2,:,i) - av_data_devide_2(:,1,:,i), 1), [3 2 1]);
    [sorted_array, sorted_indices] = sort(array);
    largest_indices = sorted_indices(end-149:end);
    %smallest_indices = sorted_indices(1:150);
    temp = zeros(size(av_data_devide_2,3),1);
    temp(largest_indices) = 1;
    %temp(smallest_indices) = 1;
    temp = reshape(temp,size(data,1),size(data,2));
    res_pixels_2(:,:,i) = temp;
end

res_R = X(:,:,1);
res_G = X(:,:,2);
res_B = X(:,:,3);
% 1,9には刺激が出ていない
for i = 2:8
    temp = res_pixels_2(:,:,i);
    res_R(temp==1) = colour(i,1);
    res_G(temp==1) = colour(i,2);
    res_B(temp==1) = colour(i,3);
end
% figure;
% imshow(cat(3,res_R,res_G,res_B))

%% 統計的に有意かつ変化も大きい

% altitude
res_R = X(:,:,1);
res_G = X(:,:,2);
res_B = X(:,:,3);

% 1,9には刺激が出ていない
for i = 2:8
    array = cell2mat(responce_pixel_num_1(i));
    temp = zeros(size(av_data_devide_1,3),1);
    temp(array(1,:)) = 1;
    temp = reshape(temp,size(data,1),size(data,2))+res_pixels_1(:,:,i);
    res_R(temp==2) = colour(i,1);
    res_G(temp==2) = colour(i,2);
    res_B(temp==2) = colour(i,3);
end
figure;
imshow(cat(3,res_R,res_G,res_B))

% azimuth
res_R = X(:,:,1);
res_G = X(:,:,2);
res_B = X(:,:,3);

% 1,9には刺激が出ていない
for i = 2:8
    array = cell2mat(responce_pixel_num_2(i));
    temp = zeros(size(av_data_devide_2,3),1);
    temp(array(1,:)) = 1;
    temp = reshape(temp,size(data,1),size(data,2))+res_pixels_2(:,:,i);
    res_R(temp==2) = colour(i,1);
    res_G(temp==2) = colour(i,2);
    res_B(temp==2) = colour(i,3);
end
figure;
imshow(cat(3,res_R,res_G,res_B))

%% altitudeとazimuthを使用してgridにする

res_R = X(:,:,1);
res_G = X(:,:,2);
res_B = X(:,:,3);

% 1,9には刺激が出ていない
for i = 4:6
    for j = 4:6
        array_1 = cell2mat(responce_pixel_num_1(i)); array_2 = cell2mat(responce_pixel_num_2(j));
        temp_1 = zeros(size(av_data_devide_1,3),1); temp_2 = zeros(size(av_data_devide_2,3),1);
        temp_1(array_1(1,:)) = 1; temp_2(array_2(1,:)) = 1;
        temp_1 = reshape(temp_1,size(data,1),size(data,2))+res_pixels_1(:,:,i);
        temp_2 = reshape(temp_2,size(data,1),size(data,2))+res_pixels_2(:,:,j);
        temp = temp_1+temp_2;
        res_R(temp==4) = colour(i-3+3*(j-4),1);
        res_G(temp==4) = colour(i-3+3*(j-4),2);
        res_B(temp==4) = colour(i-3+3*(j-4),3);
    end
end
figure;
imshow(cat(3,res_R,res_G,res_B))

%% 検定のみ

% % altitude or azimuth
% res_R = X(:,:,1);
% res_G = X(:,:,2);
% res_B = X(:,:,3);
% % 1,9には刺激が出ていない
% for i = 2:8
%     array = cell2mat(responce_pixel_num_1(i));
%     %array = cell2mat(responce_pixel_num_2(i));
%     temp = zeros(size(av_data_devide_1,3),1);
%     %temp = zeros(size(av_data_devide_1,3),1);
%     temp(array(1,:)) = 1;
%     temp = reshape(temp,size(data,1),size(data,2));
%     res_R(temp==1) = colour(i,1);
%     res_G(temp==1) = colour(i,2);
%     res_B(temp==1) = colour(i,3);
% end
% figure;
% imshow(cat(3,res_R,res_G,res_B))
% 
% % grid
% res_R = X(:,:,1);
% res_G = X(:,:,2);
% res_B = X(:,:,3);
% % 1,9には刺激が出ていない
% for i = 4:6
%     for j = 4:6
%         array_1 = cell2mat(responce_pixel_num_1(i)); array_2 = cell2mat(responce_pixel_num_2(j));
%         temp_1 = zeros(size(av_data_devide_1,3),1); temp_2 = zeros(size(av_data_devide_2,3),1);
%         temp_1(array_1(1,:)) = 1; temp_2(array_2(1,:)) = 1;
%         temp = reshape(temp_1,size(data,1),size(data,2))+reshape(temp_2,size(data,1),size(data,2));
%         res_R(temp==2) = colour(i-3+3*(j-4),1);
%         res_G(temp==2) = colour(i-3+3*(j-4),2);
%         res_B(temp==2) = colour(i-3+3*(j-4),3);
%     end
% end
% figure;
% imshow(cat(3,res_R,res_G,res_B))

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



















