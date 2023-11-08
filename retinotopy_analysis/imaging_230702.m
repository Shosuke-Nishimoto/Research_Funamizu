% imds = imageDatastore("C:\Users\funam\Desktop\analysis\imaging\230630_Blue_tiff\230630_Blue_tiff");
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
%     temp = Tiff(file_name{i}, "r");
% %     raw_data(:,:,i) = read(temp);
%     raw_data(:,:,i) = imresize(read(temp), 1/16);
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
% imshow(uint16(raw_data(:,:,5)))
% 
% %%
% save_filename = "230630_Blue_16";
% save(save_filename+".mat",'raw_data','iteNum',"-v7.3");

%%

% av = mean(raw_data,3);
% m = max(raw_data,[],3);
% ind = m>60000;
% res = av;
% res(ind) = max(m,[],"all");
% imshow(uint16(res))

%%
x_0 = 13*1920/70.71; % 32インチの横幅70.71センチ分が1920
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

image = abs(s_1 + s_2);

figure(1)
imshow(image)

%% fieldにnumbering

field_num = zeros(size(y,1), size(y,2), 9);

for i = -1:1
    for j = -1:1
        ind = theta >= (-12.5-i*25)/180*pi & theta < (-12.5+(-i+1)*25)/180*pi...
            & phi >= (-12.5+j*25)/180*pi & phi < (-12.5+(j+1)*25)/180*pi;
        temp = zeros(size(y,1), size(y,2));
        temp(ind) = 1;
        subplot(3,3,(i+2)*3-1+j);
        imshow(temp)
        field_num(:,:,(i+2)*3-1+j) = temp;
    end
end

%% 3を超えるところと0.25を超えるところをresting stateとした
load('230630_Blue_8.mat')
load('NI_30-Jun-2023_1839_26.bin.mat');
rest_duration = find(ch(7,:,:)>0.25, 1 ) - find(ch(1,:,:)>4, 1 );
rest_data = raw_data(:,:,1:rest_duration);
f = mean(rest_data,3);

%% df/fを計算
data = (raw_data-f)./f;

%%
av = mean(raw_data,3);
m = max(raw_data,[],3);
ind_high = m>max(m,[],"all")*0.85 & max(data,[],3)>0.01;
j = data(:,:,1);
j = j(ind_high);
res = av;
res(ind_high) = max(m,[],"all");
imshow(uint16(res))


%%
y = [];
for i = 1:iteNum
    tt = data(:,:,i);
    y(:,:,i) = tt(ind_high);
end

yy = reshape(y,size(y,1),iteNum,1);
figure
plot(1:iteNum, yy)

%%

Fs = 60; % sampling frequency in Hz
dt = 1/Fs; % time step in seconds
T = length(raw_data); % number of time points
t = (0:T-1)*dt; % time vector

figure
plot(t, yy)

%%
figure;
plot(ch(7,:))

%%
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

%% 1-9のfieldに刺激が出ていたかどうか

field_judge = zeros(size(field_num,3),max_stim_num);
for i = 1:max_stim_num
    temp = load("stim_"+num2str(i));
    temp = temp.n;
    for ii = 1:size(field_num,3)
        ttemp = temp+field_num(:,:,ii);
        field_judge(ii,i) = abs(isempty(find(ttemp==2, 1))-1);
    end
end

%% 実際に提示した刺激の1-9のfieldに刺激があったかどうか

field_judge_real = field_judge;
r = rem(length(image_stim_align), max_stim_num);
q = (length(image_stim_align)-r)/max_stim_num;

for i = 1:q-1
    field_judge_real = cat(2,field_judge_real,field_judge);
end
field_judge_real = cat(2,field_judge_real,field_judge(:,1:r));

%%
image_data = zeros(size(data,1)*size(data,2),size(data,3));
for i = 1:size(data,3)
    image_data(:,i) = reshape(data(:,:,i),size(data,1)*size(data,2),1);
end

figure;
plot(t, image_data)
%% 60Hzは0.01666s=17msごとのサンプリングなので、刺激提示後6 frames (約100ms分)以上とってくる
% fieldに刺激が10 frame以上でているところの最初のindexを取ってくる
% ind_field_1 = myFind(field_judge_real(:,:,1),1,10,1);
% ind_field_2 = myFind(field_judge_real(:,:,2),1,10,1);
% ind_field_3 = myFind(field_judge_real(:,:,3),1,10,1);
% ind_field_4 = myFind(field_judge_real(:,:,4),1,10,1);
% ind_field_5 = myFind(field_judge_real(:,:,5),1,10,1);
% ind_field_6 = myFind(field_judge_real(:,:,6),1,10,1);
% ind_field_7 = myFind(field_judge_real(:,:,7),1,10,1);
% ind_field_8 = myFind(field_judge_real(:,:,8),1,10,1);
% ind_field_9 = myFind(field_judge_real(:,:,9),1,10,1);

ind_field = [];
for i = 1:9
    % fieldに刺激が10 frame以上でているところの最初のindexを取ってくる
    k = myFind(field_judge_real(i,:),1,10,1);
    l = myFind(field_judge_real(i,:),0,20,1);
    for j = 1:size(l,1)-1
        temp = k > l(j) & k < l(j+1);
        ind_field(i,j) = min(k(temp));
    end
end

%%
% num_frames = 15;
% time = (-5:num_frames-6)*dt;
% time2 = [time, fliplr(time)];
num_frames = 200;
time = (-99:num_frames-100)*dt;
time2 = [time, fliplr(time)];

%%
all_data = zeros(size(image_data,1),num_frames,size(ind_field,2),9);% [ピクセル数 フレーム数 イベント数 フィールド数]
av_data = zeros(size(image_data,1),num_frames,9);
std_data = zeros(size(image_data,1),num_frames,9);

for i = 1:9
    for j = 1:num_frames
        temp = image_data(:,ind_field(i,:)+j-100);
        all_data(:,j,:,i) = temp;
        av_data(:,j,i) = mean(temp,2);
        std_data(:,j,i) = std(temp,0,2);
    end
end

figure;
plot(time, av_data(:,:,1))

figure;
curve1 = av_data(:,:,1)+std_data(:,:,1);
curve2 = av_data(:,:,1)-std_data(:,:,1);
inBetween = [curve1(1820,:), fliplr(curve2(1820,:))];
fill(time2, inBetween, 'b','FaceAlpha',.3,'EdgeAlpha',.3);
hold on;
plot(time, av_data(1820,:,1), '-b', 'LineWidth', 2);

%%
% field 1に反応したピクセル
all_data_1 = zeros(size(image_data,1),num_frames,size(ind_field_1,1));
av_data_1 = zeros(size(image_data,1),num_frames);
std_data_1 = zeros(size(image_data,1),num_frames);
for i = 1:num_frames
    all_data = image_data(:,ind_field_1+i-6);
    all_data_1(:,i,:) = all_data;
    av_data_1(:,i) = mean(all_data,2);
    std_data_1(:,i) = std(all_data,0,2);
end

figure;
plot(time, av_data_1)

figure;
curve1 = av_data_1+std_data_1;
curve2 = av_data_1-std_data_1;
inBetween = [curve1(1820,:), fliplr(curve2(1820,:))];
fill(time2, inBetween, 'b','FaceAlpha',.3,'EdgeAlpha',.3);
hold on;
plot(time, av_data_1(1820,:), '-b', 'LineWidth', 2);

%%
% onset 5 frame 前からとonset 10 frame後までで変化があったか

all_data_1_devide = permute(all_data_1,[3 2 1]);
av_data_1_devide = zeros(size(all_data_1,3),2,size(all_data_1,1));
for i = 1:size(all_data_1_devide,3)
    av_data_1_devide(:,:,i) = [mean(all_data_1_devide(:,1:5,i),2), mean(all_data_1_devide(:,6:num_frames,i),2)];
end

responce_1_pixel_num = [];
for i = 1:size(av_data_1_devide,3)
    h = ttest(av_data_1_devide(:,1,i), av_data_1_devide(:,2,i),'Alpha',0.01);
    if h==1
        responce_1_pixel_num(end+1) = i;
    end
end
figure;
boxchart(av_data_1_devide(:,:,responce_1_pixel_num(1)))

av = mean(raw_data,3);
m = max(raw_data,[],3);
temp = zeros(size(av_data_1_devide,3),1);
temp(responce_1_pixel_num) = 1;
temp = reshape(temp,size(data,1),size(data,2));
res = av;
res(temp==1) = max(m,[],"all");
figure;
imshow(uint16(res))

%%
% field 2に反応したピクセル
all_data_2 = zeros(size(image_data,1),num_frames,size(ind_field_2,1));
av_data_2 = zeros(size(image_data,1),num_frames);
std_data_2 = zeros(size(image_data,1),num_frames);
for i = 1:num_frames
    all_data = image_data(:,ind_field_2+i-6);
    all_data_2(:,i,:) = all_data;
    av_data_2(:,i) = mean(all_data,2);
    std_data_2(:,i) = std(all_data,0,2);
end

figure;
plot(time, av_data_2)

figure;
curve1 = av_data_2+std_data_2;
curve2 = av_data_2-std_data_2;
inBetween = [curve1(1820,:), fliplr(curve2(1820,:))];
fill(time2, inBetween, 'b','FaceAlpha',.3,'EdgeAlpha',.3);
hold on;
plot(time, av_data_2(1820,:), '-b', 'LineWidth', 2);

%%
% onset 5 frame 前からとonset 10 frame後までで変化があったか

all_data_2_devide = permute(all_data_2,[3 2 1]);
av_data_2_devide = zeros(size(all_data_2,3),2,size(all_data_2,1));
for i = 1:size(all_data_2_devide,3)
    av_data_2_devide(:,:,i) = [mean(all_data_2_devide(:,1:5,i),2), mean(all_data_2_devide(:,6:num_frames,i),2)];
end

responce_2_pixel_num = [];
for i = 1:size(av_data_2_devide,3)
    h = ttest(av_data_2_devide(:,1,i), av_data_2_devide(:,2,i),'Alpha',0.005);
    if h==1
        responce_2_pixel_num(end+1) = i;
    end
end
figure;
boxchart(av_data_2_devide(:,:,responce_2_pixel_num(1)))

av = mean(raw_data,3);
m = max(raw_data,[],3);
ind_high = m>max(m,[],"all")*0.85 & max(data,[],3)>0.01;
temp = zeros(size(av_data_2_devide,3),1);
temp(responce_2_pixel_num) = 1;
temp = reshape(temp,size(data,1),size(data,2));
res = av;
res(temp==1) = max(m,[],"all");
figure;
imshow(uint16(res))

%%
% field 3に反応したピクセル
all_data_3 = zeros(size(image_data,1),num_frames,size(ind_field_3,1));
av_data_3 = zeros(size(image_data,1),num_frames);
std_data_3 = zeros(size(image_data,1),num_frames);
for i = 1:num_frames
    all_data = image_data(:,ind_field_3+i-6);
    all_data_3(:,i,:) = all_data;
    av_data_3(:,i) = mean(all_data,2);
    std_data_3(:,i) = std(all_data,0,2);
end

figure;
plot(time, av_data_3)

figure;
curve1 = av_data_3+std_data_3;
curve2 = av_data_3-std_data_3;
inBetween = [curve1(1820,:), fliplr(curve2(1820,:))];
fill(time2, inBetween, 'b','FaceAlpha',.3,'EdgeAlpha',.3);
hold on;
plot(time, av_data_3(1820,:), '-b', 'LineWidth', 2);

%%
% onset 5 frame 前からとonset 10 frame後までで変化があったか

all_data_3_devide = permute(all_data_3,[3 2 1]);
av_data_3_devide = zeros(size(all_data_3,3),2,size(all_data_3,1));
for i = 1:size(all_data_3_devide,3)
    av_data_3_devide(:,:,i) = [mean(all_data_3_devide(:,1:5,i),2), mean(all_data_3_devide(:,6:num_frames,i),2)];
end

responce_3_pixel_num = [];
for i = 1:size(av_data_3_devide,3)
    h = ttest(av_data_3_devide(:,1,i), av_data_3_devide(:,2,i),'Alpha',0.005);
    if h==1
        responce_3_pixel_num(end+1) = i;
    end
end
figure;
boxchart(av_data_3_devide(:,:,responce_3_pixel_num(1)))

av = mean(raw_data,3);
m = max(raw_data,[],3);
ind_high = m>max(m,[],"all")*0.85 & max(data,[],3)>0.01;
temp = zeros(size(av_data_3_devide,3),1);
temp(responce_3_pixel_num) = 1;
temp = reshape(temp,size(data,1),size(data,2));
res = av;
res(temp==1) = max(m,[],"all");
figure;
imshow(uint16(res))

%%
% field 4に反応したピクセル
all_data_4 = zeros(size(image_data,1),num_frames,size(ind_field_4,1));
av_data_4 = zeros(size(image_data,1),num_frames);
std_data_4 = zeros(size(image_data,1),num_frames);
for i = 1:num_frames
    all_data = image_data(:,ind_field_4+i-6);
    all_data_4(:,i,:) = all_data;
    av_data_4(:,i) = mean(all_data,2);
    std_data_4(:,i) = std(all_data,0,2);
end

figure;
plot(time, av_data_4)

figure;
curve1 = av_data_4+std_data_4;
curve2 = av_data_4-std_data_4;
inBetween = [curve1(1820,:), fliplr(curve2(1820,:))];
fill(time2, inBetween, 'b','FaceAlpha',.3,'EdgeAlpha',.3);
hold on;
plot(time, av_data_4(1820,:), '-b', 'LineWidth', 2);

%%
% onset 5 frame 前からとonset 10 frame後までで変化があったか

all_data_4_devide = permute(all_data_4,[3 2 1]);
av_data_4_devide = zeros(size(all_data_4,3),2,size(all_data_4,1));
for i = 1:size(all_data_4_devide,3)
    av_data_4_devide(:,:,i) = [mean(all_data_4_devide(:,1:5,i),2), mean(all_data_4_devide(:,6:num_frames,i),2)];
end

responce_4_pixel_num = [];
for i = 1:size(av_data_4_devide,3)
    h = ttest(av_data_4_devide(:,1,i), av_data_4_devide(:,2,i),'Alpha',0.005);
    if h==1
        responce_4_pixel_num(end+1) = i;
    end
end
figure;
boxchart(av_data_4_devide(:,:,responce_4_pixel_num(1)))

av = mean(raw_data,3);
m = max(raw_data,[],3);
ind_high = m>max(m,[],"all")*0.85 & max(data,[],3)>0.01;
temp = zeros(size(av_data_4_devide,3),1);
temp(responce_4_pixel_num) = 1;
temp = reshape(temp,size(data,1),size(data,2));
res = av;
res(temp==1) = max(m,[],"all");
figure;
imshow(uint16(res))

%%
% field 5に反応したピクセル
all_data_5 = zeros(size(image_data,1),num_frames,size(ind_field_5,1));
av_data_5 = zeros(size(image_data,1),num_frames);
std_data_5 = zeros(size(image_data,1),num_frames);
for i = 1:num_frames
    all_data = image_data(:,ind_field_5+i-6);
    all_data_5(:,i,:) = all_data;
    av_data_5(:,i) = mean(all_data,2);
    std_data_5(:,i) = std(all_data,0,2);
end

figure;
plot(time, av_data_5)

figure;
curve1 = av_data_5+std_data_5;
curve2 = av_data_5-std_data_5;
inBetween = [curve1(1820,:), fliplr(curve2(1820,:))];
fill(time2, inBetween, 'b','FaceAlpha',.3,'EdgeAlpha',.3);
hold on;
plot(time, av_data_5(1820,:), '-b', 'LineWidth', 2);

%%
% onset 5 frame 前からとonset 10 frame後までで変化があったか

all_data_5_devide = permute(all_data_5,[3 2 1]);
av_data_5_devide = zeros(size(all_data_5,3),2,size(all_data_5,1));
for i = 1:size(all_data_5_devide,3)
    av_data_5_devide(:,:,i) = [mean(all_data_5_devide(:,1:5,i),2), mean(all_data_5_devide(:,6:num_frames,i),2)];
end

responce_5_pixel_num = [];
for i = 1:size(av_data_5_devide,3)
    h = ttest(av_data_5_devide(:,1,i), av_data_5_devide(:,2,i),'Alpha',0.005);
    if h==1
        responce_5_pixel_num(end+1) = i;
    end
end
figure;
boxchart(av_data_5_devide(:,:,responce_5_pixel_num(1)))

av = mean(raw_data,3);
m = max(raw_data,[],3);
ind_high = m>max(m,[],"all")*0.85 & max(data,[],3)>0.01;
temp = zeros(size(av_data_1_devide,3),1);
temp(responce_5_pixel_num) = 1;
temp = reshape(temp,size(data,1),size(data,2));
res = av;
res(temp==1) = max(m,[],"all");
figure;
imshow(uint16(res))

%%
% field 6に反応したピクセル
all_data_6 = zeros(size(image_data,1),num_frames,size(ind_field_6,1));
av_data_6 = zeros(size(image_data,1),num_frames);
std_data_6 = zeros(size(image_data,1),num_frames);
for i = 1:num_frames
    all_data = image_data(:,ind_field_6+i-6);
    all_data_6(:,i,:) = all_data;
    av_data_6(:,i) = mean(all_data,2);
    std_data_6(:,i) = std(all_data,0,2);
end

figure;
plot(time, av_data_6)

figure;
curve1 = av_data_6+std_data_6;
curve2 = av_data_6-std_data_6;
inBetween = [curve1(1820,:), fliplr(curve2(1820,:))];
fill(time2, inBetween, 'b','FaceAlpha',.3,'EdgeAlpha',.3);
hold on;
plot(time, av_data_6(1820,:), '-b', 'LineWidth', 2);

%%
% onset 5 frame 前からとonset 10 frame後までで変化があったか

all_data_6_devide = permute(all_data_6,[3 2 1]);
av_data_6_devide = zeros(size(all_data_6,3),2,size(all_data_6,1));
for i = 1:size(all_data_6_devide,3)
    av_data_6_devide(:,:,i) = [mean(all_data_6_devide(:,1:5,i),2), mean(all_data_6_devide(:,6:num_frames,i),2)];
end

responce_6_pixel_num = [];
for i = 1:size(av_data_6_devide,3)
    h = ttest(av_data_6_devide(:,1,i), av_data_6_devide(:,2,i),'Alpha',0.005);
    if h==1
        responce_6_pixel_num(end+1) = i;
    end
end
figure;
boxchart(av_data_6_devide(:,:,responce_6_pixel_num(1)))

av = mean(raw_data,3);
m = max(raw_data,[],3);
ind_high = m>max(m,[],"all")*0.85 & max(data,[],3)>0.01;
temp = zeros(size(av_data_6_devide,3),1);
temp(responce_6_pixel_num) = 1;
temp = reshape(temp,size(data,1),size(data,2));
res = av;
res(temp==1) = max(m,[],"all");
figure;
imshow(uint16(res))

%%
% field 7に反応したピクセル
all_data_7 = zeros(size(image_data,1),num_frames,size(ind_field_7,1));
av_data_7 = zeros(size(image_data,1),num_frames);
std_data_7 = zeros(size(image_data,1),num_frames);
for i = 1:num_frames
    all_data = image_data(:,ind_field_7+i-6);
    all_data_7(:,i,:) = all_data;
    av_data_7(:,i) = mean(all_data,2);
    std_data_7(:,i) = std(all_data,0,2);
end

figure;
plot(time, av_data_3)

figure;
curve1 = av_data_7+std_data_7;
curve2 = av_data_7-std_data_7;
inBetween = [curve1(1820,:), fliplr(curve2(1820,:))];
fill(time2, inBetween, 'b','FaceAlpha',.3,'EdgeAlpha',.3);
hold on;
plot(time, av_data_7(1820,:), '-b', 'LineWidth', 2);

%%
% onset 5 frame 前からとonset 10 frame後までで変化があったか

all_data_7_devide = permute(all_data_7,[3 2 1]);
av_data_7_devide = zeros(size(all_data_7,3),2,size(all_data_7,1));
for i = 1:size(all_data_7_devide,3)
    av_data_7_devide(:,:,i) = [mean(all_data_7_devide(:,1:5,i),2), mean(all_data_7_devide(:,6:num_frames,i),2)];
end

responce_7_pixel_num = [];
for i = 1:size(av_data_7_devide,3)
    h = ttest(av_data_7_devide(:,1,i), av_data_7_devide(:,2,i),'Alpha',0.005);
    if h==1
        responce_7_pixel_num(end+1) = i;
    end
end
figure;
boxchart(av_data_7_devide(:,:,responce_7_pixel_num(1)))

av = mean(raw_data,3);
m = max(raw_data,[],3);
ind_high = m>max(m,[],"all")*0.85 & max(data,[],3)>0.01;
temp = zeros(size(av_data_7_devide,3),1);
temp(responce_7_pixel_num) = 1;
temp = reshape(temp,size(data,1),size(data,2));
res = av;
res(temp==1) = max(m,[],"all");
figure;
imshow(uint16(res))

%%
% field 8に反応したピクセル
all_data_8 = zeros(size(image_data,1),num_frames,size(ind_field_8,1));
av_data_8 = zeros(size(image_data,1),num_frames);
std_data_8 = zeros(size(image_data,1),num_frames);
for i = 1:num_frames
    all_data = image_data(:,ind_field_8+i-6);
    all_data_8(:,i,:) = all_data;
    av_data_8(:,i) = mean(all_data,2);
    std_data_8(:,i) = std(all_data,0,2);
end

figure;
plot(time, av_data_8)

figure;
curve1 = av_data_8+std_data_8;
curve2 = av_data_8-std_data_8;
inBetween = [curve1(1820,:), fliplr(curve2(1820,:))];
fill(time2, inBetween, 'b','FaceAlpha',.3,'EdgeAlpha',.3);
hold on;
plot(time, av_data_8(1820,:), '-b', 'LineWidth', 2);

%%
% onset 5 frame 前からとonset 10 frame後までで変化があったか

all_data_8_devide = permute(all_data_8,[3 2 1]);
av_data_8_devide = zeros(size(all_data_8,3),2,size(all_data_8,1));
for i = 1:size(all_data_8_devide,3)
    av_data_8_devide(:,:,i) = [mean(all_data_8_devide(:,1:5,i),2), mean(all_data_8_devide(:,6:num_frames,i),2)];
end

responce_8_pixel_num = [];
for i = 1:size(av_data_8_devide,3)
    h = ttest(av_data_8_devide(:,1,i), av_data_8_devide(:,2,i),'Alpha',0.005);
    if h==1
        responce_8_pixel_num(end+1) = i;
    end
end
figure;
boxchart(av_data_8_devide(:,:,responce_8_pixel_num(1)))

av = mean(raw_data,3);
m = max(raw_data,[],3);
ind_high = m>max(m,[],"all")*0.85 & max(data,[],3)>0.01;
temp = zeros(size(av_data_8_devide,3),1);
temp(responce_8_pixel_num) = 1;
temp = reshape(temp,size(data,1),size(data,2));
res = av;
res(temp==1) = max(m,[],"all");
figure;
imshow(uint16(res))

%%
% field 9に反応したピクセル
all_data_9 = zeros(size(image_data,1),num_frames,size(ind_field_9,1));
av_data_9 = zeros(size(image_data,1),num_frames);
std_data_9 = zeros(size(image_data,1),num_frames);
for i = 1:num_frames
    all_data = image_data(:,ind_field_9+i-6);
    all_data_9(:,i,:) = all_data;
    av_data_9(:,i) = mean(all_data,2);
    std_data_9(:,i) = std(all_data,0,2);
end

figure;
plot(time, av_data_9)

figure;
curve1 = av_data_9+std_data_9;
curve2 = av_data_9-std_data_9;
inBetween = [curve1(1820,:), fliplr(curve2(1820,:))];
fill(time2, inBetween, 'b','FaceAlpha',.3,'EdgeAlpha',.3);
hold on;
plot(time, av_data_9(1820,:), '-b', 'LineWidth', 2);

%%
% onset 5 frame 前からとonset 10 frame後までで変化があったか

all_data_9_devide = permute(all_data_9,[3 2 1]);
av_data_9_devide = zeros(size(all_data_9,3),2,size(all_data_9,1));
for i = 1:size(all_data_9_devide,3)
    av_data_9_devide(:,:,i) = [mean(all_data_9_devide(:,1:5,i),2), mean(all_data_9_devide(:,6:num_frames,i),2)];
end

responce_9_pixel_num = [];
for i = 1:size(av_data_9_devide,3)
    h = ttest(av_data_9_devide(:,1,i), av_data_9_devide(:,2,i),'Alpha',0.005);
    if h==1
        responce_9_pixel_num(end+1) = i;
    end
end
figure;
boxchart(av_data_9_devide(:,:,responce_9_pixel_num(1)))

av = mean(raw_data,3);
m = max(raw_data,[],3);
ind_high = m>max(m,[],"all")*0.85 & max(data,[],3)>0.01;
temp = zeros(size(av_data_9_devide,3),1);
temp(responce_9_pixel_num) = 1;
temp = reshape(temp,size(data,1),size(data,2));
res = av;
res(temp==1) = max(m,[],"all");
figure;
imshow(uint16(res))

%%
% field 1に反応したピクセル
all_data_1 = zeros(size(image_data,1),num_frames,size(ind_field_1,1));
av_data_1 = zeros(size(image_data,1),num_frames);
std_data_1 = zeros(size(image_data,1),num_frames);
for i = 1:num_frames
    all_data = image_data(:,ind_field_1+i-100);
    all_data_1(:,i,:) = all_data;
    av_data_1(:,i) = mean(all_data,2);
    std_data_1(:,i) = std(all_data,0,2);
end

figure;
plot(time, av_data_1)

figure;
curve1 = av_data_1+std_data_1;
curve2 = av_data_1-std_data_1;
inBetween = [curve1(1820,:), fliplr(curve2(1820,:))];
fill(time2, inBetween, 'b','FaceAlpha',.3,'EdgeAlpha',.3);
hold on;
plot(time, av_data_1(1820,:), '-b', 'LineWidth', 2);

%%
% onset 99 frame 前からとonset 100 frame後までで変化があったか

all_data_1_devide = permute(all_data_1,[3 2 1]);
av_data_1_devide = zeros(size(all_data_1,3),2,size(all_data_1,1));
for i = 1:size(all_data_1_devide,3)
    av_data_1_devide(:,:,i) = [mean(all_data_1_devide(:,1:100,i),2), mean(all_data_1_devide(:,101:num_frames,i),2)];
end

responce_1_pixel_num = [];
for i = 1:size(av_data_1_devide,3)
    h = ttest(av_data_1_devide(:,1,i), av_data_1_devide(:,2,i),'Alpha',0.005);
    if h==1
        responce_1_pixel_num(end+1) = i;
    end
end
figure;
boxchart(av_data_1_devide(:,:,responce_1_pixel_num(1)))

av = mean(raw_data,3);
m = max(raw_data,[],3);
temp = zeros(size(av_data_1_devide,3),1);
temp(responce_1_pixel_num) = 1;
temp = reshape(temp,size(data,1),size(data,2));
res = av;
res(temp==1) = max(m,[],"all");
figure;
imshow(uint16(res))

figure;
plot(time, av_data_1(responce_1_pixel_num(1),:))

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



















