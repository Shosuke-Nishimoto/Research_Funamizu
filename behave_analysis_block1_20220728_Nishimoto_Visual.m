%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}

function behave_analysis_block1_20220728_Nishimoto_Visual

%block_number = 6;
[pathname1] = uigetdir
cd(pathname1)
% temp_cd = ['cd ',pathname1];
% eval(temp_cd);
tif_name = dir('block*.mat'); %get all the block files
length(tif_name)

Correct_side = [];
Chosen_side = [];
Outcome = []; %Detect free water trials or errors
EvidenceStrength = []; %Detect free water trials or errors

BlockTrial1 = [];
BlockTrial2 = [];
BlockTrial3 = [];
BlockProb1 = [];
BlockProb2 = [];
BlockProb3 = [];
Reward_LCR = [];
TrialBlock = [];
TrialCount = [];
Intensity = [];
StimDuration = [];

Trial_time = [];
count = 0;
all_trial_time = [];
trial_Tup = [];
trial_sound = []; 
trial_lick_L = []; 
trial_lick_C = [];  
trial_lick_R = [];   
InitBlock = [];

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
    Correct_side = [Correct_side; temp.saveBlock.TrueSide'];
    Chosen_side  = [Chosen_side;  temp.saveBlock.ChosenSide'];
    Outcome = [Outcome; temp.saveBlock.Outcome'];
    EvidenceStrength = [EvidenceStrength; temp.saveBlock.EvidenceStrength'];
    TrialBlock = [TrialBlock; temp.saveBlock.TrialCondition'];
    TrialCount = [TrialCount; temp.saveBlock.TrialOutcome'];
    %temp_reward = [temp.saveBlock.LeftRewardAmount', temp.saveBlock.CenterRewardAmount', temp.saveBlock.RightRewardAmount'];
    CenterRewardAmount = zeros(length(temp.saveBlock.LeftRewardAmount),1);
    temp_reward = [temp.saveBlock.LeftRewardAmount', CenterRewardAmount, temp.saveBlock.RightRewardAmount'];
    Reward_LCR = [Reward_LCR; temp_reward];
    StimDuration = [StimDuration; temp.saveBlock.StimDuration'];
    
    BlockTrial1 = [BlockTrial1; temp.saveBlock.BlockTrial1'];
    BlockTrial2 = [BlockTrial2; temp.saveBlock.BlockTrial2'];
    BlockTrial3 = [BlockTrial3; temp.saveBlock.BlockTrial3'];

    if isfield(temp.saveBlock,'InitBlock'),
        InitBlock = [InitBlock; temp.saveBlock.InitBlock'];
    end
    if isfield(temp.saveBlock,'BlockProb1'),
        BlockProb1 = [BlockProb1; temp.saveBlock.BlockProb1'];
        BlockProb2 = [BlockProb2; temp.saveBlock.BlockProb2'];
        if isfield(temp.saveBlock,'BlockProb3')
            BlockProb3 = [BlockProb3; temp.saveBlock.BlockProb3'];
        else
            BlockProb3 = 0.5;
        end
    else
        BlockProb1 = 0.5;
        BlockProb2 = 0.5;
        BlockProb3 = 0.5;
    end
    
    temp_start = temp.saveBlock.TrialStart';
    temp_end   = temp.saveBlock.TrialEnd';
    temp_start = [temp_start, temp_end];
    Trial_time = [Trial_time; temp_start];
    
    temp_trial = temp.saveBlock.RawEvents.Trial;
    for j = 1:length(temp_trial),
        %j
        clear trial_time
        count = count + 1;
        temp_temp = temp_trial{j};
        temp_states = temp_temp.States;
        temp_events = temp_temp.Events;
        
        %Get the time for state
        trial_time(1).matrix = temp_states.Timerset;
        trial_time(2).matrix = temp_states.WaitLedOn;
        trial_time(3).matrix = temp_states.WaitSoundOn;
        trial_time(4).matrix = temp_states.TriggerStim;
        trial_time(5).matrix = temp_states.Stim;
        trial_time(6).matrix = temp_states.Hold;
        trial_time(7).matrix = temp_states.SpoutGo;
        trial_time(8).matrix = temp_states.Choice;
        trial_time(9).matrix = temp_states.Reward_wait;
        trial_time(10).matrix = temp_states.Reward;
        trial_time(11).matrix = temp_states.Reward_after;
        trial_time(12).matrix = temp_states.Reward_ITI;
        trial_time(13).matrix = temp_states.IC;
        trial_time(14).matrix = temp_states.Punish;
        trial_time(15).matrix = temp_states.Punish_after;
        trial_time(16).matrix = temp_states.PunishITI;
        trial_time(17).matrix = temp_states.NoResponse;
        trial_time(18).matrix = temp_states.EndState;
        trial_time(19).matrix = temp_states.NoStim;
        
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
        %if isfield(temp_events,'Port1In'),
        if isfield(temp_events,'Port1In') && isfield(temp_events,'Port1Out'),
            % lick_L = [temp_events.Port1In', temp_events.Port1Out'];
            %Check the Port2 lick
            lick_in  = temp_events.Port1In';
            lick_out = temp_events.Port1Out';
            %More than one difference is error
            if abs(length(lick_in) - length(lick_out)) > 1,
                hoge
            end
            lick_out = [lick_out; max(Tup)];
            for k = 1:length(lick_in),
                temp_out = find(lick_out > lick_in(k), 1);
                lick_L(k,:) = [lick_in(k), lick_out(temp_out)];
            end
        else
            lick_L = [];
        end
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
        %if isfield(temp_events,'Port3In'),
        if isfield(temp_events,'Port3In') && isfield(temp_events,'Port3Out'),
            %lick_R = [temp_events.Port3In', temp_events.Port3Out'];
            %Check the Port2 lick
            lick_in  = temp_events.Port3In';
            lick_out = temp_events.Port3Out';
            %More than one difference is error
            if abs(length(lick_in) - length(lick_out)) > 1,
                hoge
            end
            lick_out = [lick_out; max(Tup)];
            for k = 1:length(lick_in),
                temp_out = find(lick_out > lick_in(k), 1);
                lick_R(k,:) = [lick_in(k), lick_out(temp_out)];
            end
        else
            lick_R = [];
        end
        
        %save trial data
        all_trial_time(count).matrix = trial_time;
        trial_Tup(count).matrix = Tup;
        trial_sound(count).matrix = sound; 
        trial_lick_L(count).matrix = lick_L; 
        trial_lick_C(count).matrix = lick_C;  
        trial_lick_R(count).matrix = lick_R;
        
        %Save tone intensity
%        Intensity(count) = temp.saveBlock.StimulusSettings{j}.Volume;
        
        %Tone cloud
        %Tone_cloud(count,:) = temp.saveBlock.Cloud{j};
%        Tone_cloud(count).matrix = temp.saveBlock.Cloud{j};
    end
%     if i == 3,
%         eg_sound = temp.saveBlock.Cloud{9};
%         eg_sound_dif = temp.saveBlock.EvidenceStrength(9);
%         temp.saveBlock.EvidenceStrength([1:50])
%     end
end
Intensity = Intensity';

% Checking block
if min(BlockTrial1) ~= max(BlockTrial1), %Change detection
    hoge
end
if min(BlockTrial2) ~= max(BlockTrial2), %Change detection
    hoge
end
if min(BlockTrial3) ~= max(BlockTrial3), %Change detection
    hoge
end
if min(BlockProb1) ~= max(BlockProb1), %Change detection
    hoge
end
if min(BlockProb2) ~= max(BlockProb2), %Change detection
    hoge
end
if min(BlockProb3) ~= max(BlockProb3), %Change detection
    hoge
end
clear BlockTrial BlockProb BlockReward
% BlockTrial(1) = max(BlockTrial1);
% BlockTrial(2) = max(BlockTrial2);
% BlockTrial(3) = max(BlockTrial3);
% BlockProb(1) = max(BlockProb1);
% BlockProb(2) = max(BlockProb2);
% BlockProb(3) = max(BlockProb3);
BlockTrial(1) = mode(BlockTrial1);
BlockTrial(2) = mode(BlockTrial2);
BlockTrial(3) = mode(BlockTrial3);
BlockProb(1) = mode(BlockProb1);
BlockProb(2) = mode(BlockProb2);
BlockProb(3) = mode(BlockProb3);
%Get Block Reward
%Reward_LCR
%TrialBlock
% for i = 1:3,
%     temp = find(TrialBlock == i);
%     temp = Reward_LCR(temp,:);
%     %Check the amount
%     min_temp = min(temp);
%     max_temp = max(temp);
%     max_temp = min_temp ~= max_temp; %Detect difference
% %     if max(max_temp),
% %         i
% %         %temp
% %         disp('Difference in reward amount')
% %         [min_temp, max_temp]
% %         %if i ~= 1
% %             hoge
% %         %end
% %     end
%     BlockReward(i,:) = [min_temp(1),min_temp(3)];
% end

if ~isempty(InitBlock)
    if min(InitBlock) ~= max(InitBlock) %Change detection
        hoge
    else
        InitBlock = mode(InitBlock);
    end
end

save_filename = ['Bpod_mat_210202_',temp_date,'_',temp_session]
%save block_mat_170222
%all_trial_time trial_Tup trial_sound trial_lick_L
%trial_lick_C trial_lick_R Correct_side Chosen_side Outcome
%EvidenceStrength Trial_time TrialBlock TrialCount 
%BlockTrial BlockProb BlockReward Reward_LCR Intensity
%save block_mat_170727 all_trial_time trial_Tup trial_sound trial_lick_L trial_lick_C trial_lick_R Correct_side Chosen_side Outcome EvidenceStrength Trial_time TrialBlock TrialCount BlockTrial BlockProb BlockReward Reward_LCR Intensity Tone_cloud
save(save_filename, 'all_trial_time', 'trial_Tup', 'trial_sound', ...
    'trial_lick_L', 'trial_lick_C', 'trial_lick_R', 'StimDuration',...
    'Correct_side', 'Chosen_side', 'Outcome', 'EvidenceStrength', ...
    'Trial_time', 'TrialBlock', 'TrialCount', ...
    'BlockTrial', 'Reward_LCR', 'InitBlock')

% save(save_filename, 'all_trial_time', 'trial_Tup', 'trial_sound', ...
%     'trial_lick_L', 'trial_lick_C', 'trial_lick_R', 'StimDuration',...
%     'Correct_side', 'Chosen_side', 'Outcome', 'EvidenceStrength', ...
%     'Trial_time', 'TrialBlock', 'TrialCount', ...
%     'BlockTrial', 'BlockProb', 'BlockReward', ...
%     'Reward_LCR', 'Intensity', 'Tone_cloud', 'InitBlock')

% %Plot sound,
% %78 sound component
% eg_sound_dif
% figure
% for i = 1:length(eg_sound),
%     plot([i,i+2],[eg_sound(i),eg_sound(i)],'b-')
%     hold on
% end
% %set(gca,'xlim',[-1,61],'ylim',[0 19])
% set(gca,'xlim',[-1,101],'ylim',[0 19])
% 
% hoge

