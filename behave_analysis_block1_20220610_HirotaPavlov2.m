%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}

function behave_analysis_block1_20220610_HirotaPavlov2

[filename1, pathname1]=uigetfile('*.mat','Block_mat');
filename1 = [pathname1, filename1];
load(filename1)

% %all_trial_time trial_Tup trial_sound trial_lick_L
% %trial_lick_C trial_lick_R 
% %Correct_side Chosen_side Outcome
% %EvidenceStrength Trial_time TrialBlock TrialCount 
% %BlockTrial BlockProb BlockReward Reward_LCR Intensity
% %save block_mat_170727 all_trial_time trial_Tup trial_sound trial_lick_L trial_lick_C trial_lick_R Correct_side Chosen_side Outcome EvidenceStrength Trial_time TrialBlock TrialCount BlockTrial BlockProb BlockReward Reward_LCR Intensity Tone_cloud
% save(save_filename, 'all_trial_time', 'trial_Tup', 'trial_sound', ...
%     'trial_lick_L', 'trial_lick_C', 'trial_lick_R', 'Trial_time', ...
%     'TimeForTrial', 'ITI', 'ToneDuration', 'VisualDuration', ...
%     'PreStimDuration', 'HoldDuration', 'VisionOnOff', 'FreqSide', ...
%      'SoundFreq', 'Transition', 'RewardInfo', 'UseFreq', ...
%     'UseVolume', 'Reward_prob', 'RewardAmount', 'AccumulatedWater', ...
%     'VisionSound')

Ntrial = length(all_trial_time);
vision1 = find(VisionSound(:,2) == 1);
vision2 = find(VisionSound(:,2) == 2);

[unique(ToneDuration),unique(VisualDuration)]
[length(vision1),length(vision2)]

time_window = [-500:10:3200];

%Make Gaussian filters with the STD of 10 ms
%Make the time windows (one window = 10 ms)
gauss_std = 10; %now 100ms std
width = 10; %100ms std
for i = 1:length(time_window)-1
    use_time = [i-width : i + width];
    temp0 = find(use_time > 0 & use_time <= length(time_window)-1);
    use_time = use_time(temp0);
    
    gauss_pdf = normpdf(use_time,i,gauss_std);
    gauss_pdf = gauss_pdf ./ sum(gauss_pdf);
    SDF_filter(i).time = use_time;
    SDF_filter(i).gauss_pdf = gauss_pdf;
end

[N_lick1,window_lick1,spike_filter1] = plot_center_licks(vision1,trial_sound,trial_lick_C,time_window,SDF_filter);
[N_lick2,window_lick2,spike_filter2] = plot_center_licks(vision2,trial_sound,trial_lick_C,time_window,SDF_filter);

figure
%plot(mean(window_lick1),'b')
plot(mean(spike_filter1),'b')
%plot_mean_se_moto(spike_filter1,[0 0 1],2)
hold on
%plot(mean(window_lick2),'r')
plot(mean(spike_filter2),'r')
%plot_mean_se_moto(spike_filter2,[1 0 0],2)

ranksum(N_lick1,N_lick2)
[median(N_lick1), median(N_lick2)]
[mean(N_lick1), mean(N_lick2)]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [N_lick,window_lick,spike_filter] = plot_center_licks(vision1,trial_sound,trial_lick_C,time_window,SDF_filter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

window_lick = zeros(length(vision1),length(time_window)-1); %-300 ms to 3200 ms
spike_filter = zeros(length(vision1),length(time_window)-1); %-300 ms to 3200 ms

figure
for i = 1:length(vision1)
    trial = vision1(i);
    temp_sound = trial_sound(trial).matrix;
    if length(temp_sound) == 0
        hoge
    else
        temp_sound = temp_sound(1); %Sound start time
    end
    lick = trial_lick_C(trial).matrix;
    lick = lick - temp_sound;
    
    temp_x = ones(length(lick),1) * i;
    plot(lick,temp_x,'b.')
    hold on
    
    %Number of lick during 2 to 3 sec
    N_lick(i) = length(find(lick >= 2 & lick <= 3));
    
    for j = 1:length(time_window)-1
        temp = find(lick >= time_window(j)./1000 & lick < time_window(j+1)./1000);
        if ~isempty(temp)
            window_lick(i,j) = 1;
        end
    end
    
    for j = 1:length(time_window)-1
        spike_filter(i,j) = sum(window_lick(i,SDF_filter(j).time) .* SDF_filter(j).gauss_pdf);
    end
end
set(gca,'xlim',[-2 5])
return