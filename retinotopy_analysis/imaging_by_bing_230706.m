% Load the data
load('230630_Blue.mat');

%%

% data = reshape(raw_data, 512*512,length(raw_data),1);
% % Define some parameters
% Fs = 60; % sampling frequency in Hz
% dt = 1/Fs; % time step in seconds
% T = length(raw_data); % number of time points
% t = (0:T-1)*dt; % time vector
% 
% % Plot the raw data
% figure;
% plot(t,data(1:100,:));
% xlabel('Time (s)');
% ylabel('Fluorescence (a.u.)');
% title('Raw calcium imaging data');

%%

load('NI_30-Jun-2023_1839_26.bin.mat');
rest_duration = min(find(ch(7,:,:)>0.25)) - min(find(ch(1,:,:)>3));
rest_data = raw_data(:,:,1:rest_duration);
f = mean(rest_data,3);

Fs = 60; % sampling frequency in Hz
dt = 1/Fs; % time step in seconds
T = length(raw_data); % number of time points
t = (0:T-1)*dt; % time vector

%% df/fを計算
data = (raw_data-f)./f;

%%
m = max(raw_data,[],3);
ind_high = m>max(m,[],"all")*0.85 & max(data,[],3)>0.01;
j = data(:,:,1);
j = j(ind_high);
res = mean(raw_data,3);
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
plot(t, yy)
%%

% Apply a low-pass filter to remove high-frequency noise
fc = 1; % cutoff frequency in Hz
[b,a] = butter(2,fc/(Fs/2)); % design a second-order Butterworth filter
data_filt = filtfilt(b,a,yy); % apply the filter to the data

%%
% Plot the filtered data
figure;
plot(t,data_filt);
xlabel('Time (s)');
ylabel('Fluorescence (a.u.)');
title('Filtered calcium imaging data');


%%
% Detect peaks in the filtered data using a threshold
thresh = 0.1; % threshold value
[pks,locs] = findpeaks(data_filt(2300,:),'MinPeakHeight',thresh); % find peaks and their locations

% Plot the peaks on the filtered data
figure;
plot(t,data_filt(2300,:));
hold on;
plot(t(locs),pks,'ro'); % plot peaks as red circles
hold off;
xlabel('Time (s)');
ylabel('Fluorescence (a.u.)');
title('Detected peaks in calcium imaging data');

%%

% Calculate the inter-spike intervals (ISIs) and the firing rate
ISIs = diff(t(locs)); % compute the ISIs in seconds
firing_rate = 1./ISIs; % compute the firing rate in Hz

% Plot the ISIs and the firing rate as histograms
figure;
subplot(2,1,1);
histogram(ISIs,20); % plot a histogram of ISIs with 20 bins
xlabel('Inter-spike interval (s)');
ylabel('Count');
title('Histogram of inter-spike intervals');
subplot(2,1,2);
histogram(firing_rate,20); % plot a histogram of firing rate with 20 bins
xlabel('Firing rate (Hz)');
ylabel('Count');
title('Histogram of firing rate');
