%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}

function behave_analysis_block1_20220728_Nishimoto_Visual2(filename1)

switch nargin
    case 0
        [filename1, pathname1]=uigetfile('*.mat','Block_mat');
        %[filename1, pathname1]=uigetfile('Bpod*.mat','Block_mat');
        filename1 = [pathname1, filename1];
        load(filename1)
    case 1
        load(filename1)
    otherwise
        hoge
end

Choice_trial = find(Outcome == 1 | Outcome == 2);
%Choice_trial = find(Outcome == 1 | Outcome == 2 | Outcome == 4);

length(Choice_trial)
temp_evi = unique(EvidenceStrength)

Chosen_trial = Chosen_side(Choice_trial);
Correct_trial = Correct_side(Choice_trial);

temp_correct = Chosen_trial == Correct_trial;
sum(temp_correct)
length(temp_correct)

sum(temp_correct) ./ length(temp_correct)

left_correct  = find(Correct_trial == 0);
right_correct = find(Correct_trial == 1);

left_choice = Chosen_trial(left_correct);
right_choice = Chosen_trial(right_correct);

ave_choice(1) = sum(left_choice) ./ length(left_choice);
ave_choice(2) = sum(right_choice) ./ length(right_choice);

[p_Rmin(1),conf_Rmin(1,:)] = binofit(sum(left_choice),length(left_choice));
[p_Rmin(2),conf_Rmin(2,:)] = binofit(sum(right_choice),length(right_choice));

figure
subplot(1,2,1)
bar(ave_choice)
set(gca,'ylim',[0 1])
subplot(1,2,2)
plot([1,2],[p_Rmin],'b.')
hold on
plot([1,1],conf_Rmin(1,:),'k')
hold on
plot([2,2],conf_Rmin(2,:),'k')
set(gca,'xlim',[0.5 2.5],'ylim',[0 1])

ave_choice
p_Rmin
hoge

% %Outcome
% outcome_EW     = 0; %early withdrawal
% outcome_IC     = 1; %incorrect choice
% outcome_reward = 2; %reward was dispensed (either automatically in early training, or after correct choice)
% outcome_NC     = 3; %no choice was made and time elapsed
% outcome_UN     = 4; %undefined or Free water:

temp_evi = unique(EvidenceStrength);
temp_evi_low  = 0.5 - temp_evi/2;
temp_evi_high = 0.5 + temp_evi/2;
temp_evi_all = [temp_evi_low', temp_evi_high'];
tone_evidence = sort(temp_evi_all);

%Put tone evidence in all trials;
trial_evidence = zeros(length(Outcome),1);
low  = find(Correct_side == 0);
high = find(Correct_side == 1);
temp = Chosen_side == Correct_side;
correct = find(temp == 1);
error   = find(temp == 0);
for i = 1:length(temp_evi),
    temp = find(EvidenceStrength == temp_evi(i));
    temp_left  = intersect(temp,low);
    temp_right = intersect(temp,high);
    trial_evidence(temp_left)  = temp_evi_low(i);
    trial_evidence(temp_right) = temp_evi_high(i);
end

%Make the true tone cloud value
for i = 1:length(Tone_cloud)
    temp_tone = Tone_cloud(i).matrix;
    %Get the data in all sound
    temp0 = find(temp_tone <= 8);
    temp1 = find(temp_tone >= 9);
    %binary_tone(i,1) = (length(temp1)-length(temp0)) ./ length(temp_tone);
    binary_tone(i,1) = length(temp1) ./ length(temp_tone);
end


%TrialBlock
for i = 1:max(TrialBlock)
    block(i).matrix = find(TrialBlock == i);
    
    temp_block = intersect(block(i).matrix, Choice_trial);
    number_use_trial(i) = length(temp_block);
end
block2 = [];
block3 = [];
for i = 2:max(TrialBlock)
    if rem(i,2) == 0
        block2 = [block2; block(i).matrix];
    else
        block3 = [block3; block(i).matrix];
    end
end
block2 = sort(block2);
block3 = sort(block3);
%block2 = sort([block(2).matrix; block(4).matrix]);
%block3 = sort([block(3).matrix; block(5).matrix]);
use_trial2 = intersect(block2, Choice_trial);
use_trial3 = intersect(block3, Choice_trial);

%Update the use_trial2 and use_trial3
use_trial_all = [use_trial2; use_trial3];
use_trial_all = sort(use_trial_all);

min_trial2 = intersect(minD_trial, use_trial2);
min_trial3 = intersect(minD_trial, use_trial3);
min_trial_all = [min_trial2; min_trial3];
min_trial_all = sort(min_trial_all);
max_trial2 = intersect(maxD_trial, use_trial2);
max_trial3 = intersect(maxD_trial, use_trial3);
max_trial_all = [max_trial2; max_trial3];
max_trial_all = sort(max_trial_all);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Analyzing more data

%Based on the correct trial, flip the tone cloud
temp = find(Correct_side == 1); %Right is correct
if mean(binary_tone(temp)) < 0.5 %low for right correct
    disp('flip tones')
    binary_tone = 1 - binary_tone;
    flip_tone = 1;
else
    flip_tone = 0;
end

%Based on the binary tone decide the pseudo tone evidence
clear temp_tone
temp_tone(1).matrix = find(binary_tone == 0);
temp_tone(2).matrix = find(binary_tone > 0 & binary_tone <= 0.35);
temp_tone(3).matrix = find(binary_tone > 0.35 & binary_tone < 0.5);
temp_tone(4).matrix = find(binary_tone > 0.5 & binary_tone < 0.65);
temp_tone(5).matrix = find(binary_tone >= 0.65 & binary_tone < 1);
temp_tone(6).matrix = find(binary_tone == 1);

new_tone_evi = nan(length(binary_tone),1);
for i = 1:6
    new_tone_evi(temp_tone(i).matrix) = tone_evidence(i);
end
%Check nan
temp = unique(isnan(new_tone_evi));
temp = find(temp == 1);
if ~isempty(temp)
    [binary_tone, new_tone_evi]
    disp('nan detected')
    hoge
end
trial_evidence = new_tone_evi;

%Get the number of right choice trials in each session
if BlockReward(2,1) < BlockReward(2,2) % Right -> Left
    [right_trial.Rmin, number_trial.Rmin] = get_right_choice_trials(min_trial2,Chosen_side,trial_evidence,tone_evidence);
    [right_trial.Rmax, number_trial.Rmax] = get_right_choice_trials(max_trial2,Chosen_side,trial_evidence,tone_evidence);
    
    [right_trial.Lmin, number_trial.Lmin] = get_right_choice_trials(min_trial3,Chosen_side,trial_evidence,tone_evidence);
    [right_trial.Lmax, number_trial.Lmax] = get_right_choice_trials(max_trial3,Chosen_side,trial_evidence,tone_evidence);
else % Left -> Right
    [right_trial.Lmin, number_trial.Lmin] = get_right_choice_trials(min_trial2,Chosen_side,trial_evidence,tone_evidence);
    [right_trial.Lmax, number_trial.Lmax] = get_right_choice_trials(max_trial2,Chosen_side,trial_evidence,tone_evidence);
    
    [right_trial.Rmin, number_trial.Rmin] = get_right_choice_trials(min_trial3,Chosen_side,trial_evidence,tone_evidence);
    [right_trial.Rmax, number_trial.Rmax] = get_right_choice_trials(max_trial3,Chosen_side,trial_evidence,tone_evidence);
end
    
right_trial_all = right_trial.Rmin + right_trial.Rmax + right_trial.Lmin + right_trial.Lmax;
number_trial_all = number_trial.Rmin + number_trial.Rmax + number_trial.Lmin + number_trial.Lmax;

%right_trial_all ./ number_trial_all %Right choice prob

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [right_trial, number_trial] = get_right_choice_trials(use_trials,Chosen_side,trial_evidence,tone_evidence)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

use_choice = Chosen_side(use_trials);
use_tone = trial_evidence(use_trials);

for i = 1:length(tone_evidence)
    temp_trial = find(use_tone == tone_evidence(i));
    temp_choice = use_choice(temp_trial);
    number_trial(i) = length(temp_trial);
    right_trial(i) = sum(temp_choice);
end

return



