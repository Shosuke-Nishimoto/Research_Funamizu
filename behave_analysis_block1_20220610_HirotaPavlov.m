%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}

function behave_analysis_block1_20220610_HirotaPavlov

%block_number = 6;
[pathname1] = uigetdir
cd(pathname1)
% temp_cd = ['cd ',pathname1];
% eval(temp_cd);
tif_name = dir('block*.mat'); %get all the block files
length(tif_name)

TimeForTrial = []; %Time for each trial
ITI = []; %Time for stop licking before trial start
ToneDuration = []; %Tone stim duration
VisualDuration = []; %Visual stim duration
PreStimDuration = []; 
HoldDuration = [];
%RETimeoutDuration = [];
VisionOnOff = []; %Left stim reward OR Right stim reward
FreqSide = []; %Sound freq relation for reward
AccumulatedWater = [];
%Renzoku = [];

SoundFreq = []; %Sound frequencies of 4 stim {}
%NumberSound = []; %How many stimulus do you use?
Transition = []; %Transition prob from sound to vision
RewardInfo = []; %Reward info including the  {}
UseFreq = []; %Important!!, current tone freq
UseVolume = []; %current tone intensity
%VisionSide = []; %Important!!, current vision side
Reward_prob = []; %(Important!!), relative proportion of reward amount get reward
RewardAmount = []; %Important!! Reward amount
%StimulusSettings = [];
%Outcome = []; %state check
%TrialStart = []; %trial start time
%TrialEnd = []; %trial end time

%Outcome_sound = []; %Important!! Tone setting
%Outcome_vision = []; %Important!! Vision setting
VisionSound = []; %Combination of vision and sound
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Intensity = [];
Trial_time = [];
count = 0;
all_trial_time = [];
trial_Tup = [];
trial_sound = []; 
trial_lick_L = []; 
trial_lick_C = [];  
trial_lick_R = [];   
%InitBlock = [];

%Load all block data
%for i = 1 : block_number,
for i = 1 : length(tif_name),
    clear temp temp_filename
    temp_filename = sprintf('block%d',i);
    temp_filename
    %temp_filename = tif_name(i).name
    temp = load(temp_filename);
    
    temp_date = temp.saveBlock.Date;
    temp_session = temp.saveBlock.SessionDescription;
    TimeForTrial = [TimeForTrial; temp.saveBlock.TimeForTrial'];
    ITI  = [ITI;  temp.saveBlock.ITI'];
    ToneDuration = [ToneDuration; temp.saveBlock.ToneDuration'];
    VisualDuration = [VisualDuration; temp.saveBlock.VisualDuration'];
    PreStimDuration = [PreStimDuration; temp.saveBlock.PreStimDuration'];
    HoldDuration = [HoldDuration; temp.saveBlock.HoldDuration'];
    VisionOnOff = [VisionOnOff; temp.saveBlock.VisionOnOff'];
    FreqSide = [FreqSide; temp.saveBlock.FreqSide'];
    Transition = [Transition; temp.saveBlock.Transition'];
    UseFreq = [UseFreq; temp.saveBlock.UseFreq'];
    UseVolume = [UseVolume; temp.saveBlock.UseVolume'];
    Reward_prob = [Reward_prob; temp.saveBlock.Reward_prob'];
    RewardAmount = [RewardAmount; temp.saveBlock.RewardAmount'];
    AccumulatedWater = [AccumulatedWater;  temp.saveBlock.AccumulatedWater'];
    
    temp_VisionSound = [temp.saveBlock.Outcome_sound', temp.saveBlock.Outcome_vision'];
    VisionSound = [VisionSound; temp_VisionSound];
    
    size_trial = length(temp.saveBlock.SoundFreq);
    tempSF = [];
    tempR = [];
    for j = 1:size_trial
        tempSF(j,:) = temp.saveBlock.SoundFreq{j};
        tempR(j,:) = temp.saveBlock.RewardInfo{j};
    end
    SoundFreq = [SoundFreq; tempSF];
    RewardInfo = [RewardInfo; tempR];
    
    temp_start = temp.saveBlock.TrialStart';
    temp_end   = temp.saveBlock.TrialEnd';
    temp_start = [temp_start, temp_end];
    Trial_time = [Trial_time; temp_start];
    
    temp_trial = temp.saveBlock.RawEvents.Trial;
    for j = 1:length(temp_trial)
        %j
        clear trial_time
        count = count + 1;
        temp_temp = temp_trial{j};
        temp_states = temp_temp.States;
        temp_events = temp_temp.Events;
        
        %Get the time for state
        trial_time(1).matrix = temp_states.Timerset;
        trial_time(2).matrix = temp_states.WaitITI;
        trial_time(3).matrix = temp_states.BeforeNoLick;
        trial_time(4).matrix = temp_states.NoLickCheck;
        trial_time(5).matrix = temp_states.TriggerStim;
        trial_time(6).matrix = temp_states.Stim;
        trial_time(7).matrix = temp_states.PreVision;
        trial_time(8).matrix = temp_states.Vision;
        trial_time(9).matrix = temp_states.Reward_wait;
        trial_time(10).matrix = temp_states.Reward;
        trial_time(11).matrix = temp_states.Reward_ITI;
        trial_time(12).matrix = temp_states.EndState;
        trial_time(13).matrix = temp_states.NoResponse;
        trial_time(14).matrix = temp_states.NoStim;
        
        clear Tup sound lick_L lick_C lick_R
        %Get Events
        Tup = temp_events.Tup';
        %Sound inputs
        if isfield(temp_events,'BNC1High'),
            sound = [temp_events.BNC1High', temp_events.BNC1Low'];
        else
            sound = [];
        end
        %Lick
        lick_L = [];
        lick_R = [];
%         %if isfield(temp_events,'Port2In'),
%         if isfield(temp_events,'Port1In') && isfield(temp_events,'Port1Out'),
%             % lick_L = [temp_events.Port1In', temp_events.Port1Out'];
%             %Check the Port2 lick
%             lick_in  = temp_events.Port1In';
%             lick_out = temp_events.Port1Out';
%             %More than one difference is error
%             if abs(length(lick_in) - length(lick_out)) > 1,
%                 hoge
%             end
%             lick_out = [lick_out; max(Tup)];
%             for k = 1:length(lick_in),
%                 temp_out = find(lick_out > lick_in(k), 1);
%                 lick_L(k,:) = [lick_in(k), lick_out(temp_out)];
%             end
%         else
%             lick_L = [];
%         end
        %if isfield(temp_events,'Port2In'),
        if isfield(temp_events,'Port2In') && isfield(temp_events,'Port2Out'),
            %Check the Port2 lick
            lick_in  = temp_events.Port2In';
            lick_out = temp_events.Port2Out';
            %More than one difference is error
            if abs(length(lick_in) - length(lick_out)) > 1,
                hoge
            end
            lick_out = [lick_out; max(Tup)];
            for k = 1:length(lick_in),
                temp_out = find(lick_out > lick_in(k), 1);
                lick_C(k,:) = [lick_in(k), lick_out(temp_out)];
            end
            % lick_C
        else
            lick_C = [];
        end
%         %if isfield(temp_events,'Port3In'),
%         if isfield(temp_events,'Port3In') && isfield(temp_events,'Port3Out'),
%             %lick_R = [temp_events.Port3In', temp_events.Port3Out'];
%             %Check the Port2 lick
%             lick_in  = temp_events.Port3In';
%             lick_out = temp_events.Port3Out';
%             %More than one difference is error
%             if abs(length(lick_in) - length(lick_out)) > 1,
%                 hoge
%             end
%             lick_out = [lick_out; max(Tup)];
%             for k = 1:length(lick_in),
%                 temp_out = find(lick_out > lick_in(k), 1);
%                 lick_R(k,:) = [lick_in(k), lick_out(temp_out)];
%             end
%         else
%             lick_R = [];
%         end
        
        %save trial data
        all_trial_time(count).matrix = trial_time;
        trial_Tup(count).matrix = Tup;
        trial_sound(count).matrix = sound; 
        trial_lick_L(count).matrix = lick_L; 
        trial_lick_C(count).matrix = lick_C;  
        trial_lick_R(count).matrix = lick_R;
        
        %Save tone intensity
        Intensity(count) = temp.saveBlock.StimulusSettings{j}.Volume;
        
        %Tone cloud
        %Tone_cloud(count,:) = temp.saveBlock.Cloud{j};
        %Tone_cloud(count).matrix = temp.saveBlock.Cloud{j};
    end
%     if i == 3,
%         eg_sound = temp.saveBlock.Cloud{9};
%         eg_sound_dif = temp.saveBlock.EvidenceStrength(9);
%         temp.saveBlock.EvidenceStrength([1:50])
%     end
end
Intensity = Intensity';
%Intensity check
temp = Intensity == UseVolume;
if min(temp) == 0
    [Intensity, UseVolume];
    hoge
end

save_filename = ['Bpod_mat_220610_',temp_date,'_',temp_session]
%all_trial_time trial_Tup trial_sound trial_lick_L
%trial_lick_C trial_lick_R 
%Correct_side Chosen_side Outcome
%EvidenceStrength Trial_time TrialBlock TrialCount 
%BlockTrial BlockProb BlockReward Reward_LCR Intensity
%save block_mat_170727 all_trial_time trial_Tup trial_sound trial_lick_L trial_lick_C trial_lick_R Correct_side Chosen_side Outcome EvidenceStrength Trial_time TrialBlock TrialCount BlockTrial BlockProb BlockReward Reward_LCR Intensity Tone_cloud
save(save_filename, 'all_trial_time', 'trial_Tup', 'trial_sound', ...
    'trial_lick_L', 'trial_lick_C', 'trial_lick_R', 'Trial_time', ...
    'TimeForTrial', 'ITI', 'ToneDuration', 'VisualDuration', ...
    'PreStimDuration', 'HoldDuration', 'VisionOnOff', 'FreqSide', ...
     'SoundFreq', 'Transition', 'RewardInfo', 'UseFreq', ...
    'UseVolume', 'Reward_prob', 'RewardAmount', 'AccumulatedWater', ...
    'VisionSound')