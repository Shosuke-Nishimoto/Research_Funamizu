
%{
----------------------------------------------------------------------------
Determine the time window for task relevant neurons
%Start of task
%Sound on
%Sound off
%Before choice
%After choice (0sec)
%After choice (1sec)
%After choice (2sec)
%p = 0.001
%Each epoch, predict the prior, sensory and choice (integration)
----------------------------------------------------------------------------
%}

function process_20231002_encoding_model_compare_depth_control

[auc_sound,auc_choice,auc_prior, auc_prob, auc_neuron] = process_20231002_encoding_model2_depth('auc_20220820');
[fof_sound,fof_choice,fof_prior, fof_prob, fof_neuron] = process_20231002_encoding_model2_depth('fof_20220820');
[mpfc_sound,mpfc_choice,mpfc_prior, mpfc_prob, mpfc_neuron] = process_20231002_encoding_model2_depth('mpfc_20220820');
close all

[p_sound,median_sound] = ranksum_areas(auc_sound,fof_sound,mpfc_sound);
[p_choice,median_choice] = ranksum_areas(auc_choice,fof_choice,mpfc_choice);
[p_prior,median_prior] = ranksum_areas(auc_prior,fof_prior,mpfc_prior);

median_sound
p_sound(1)
p_sound(2)
p_sound(3)

median_choice
p_choice(1)
p_choice(2)
p_choice(3)

median_prior
p_prior(1)
p_prior(2)
p_prior(3)

[auc_neuron(4),fof_neuron(4),mpfc_neuron(4)]

figure
bar([auc_prob;fof_prob;mpfc_prob]')

%Significant test with kai 2 test
%sound, choice, prior 
p_sound = kai2test_3regions(auc_neuron,fof_neuron,mpfc_neuron,1);
p_choice = kai2test_3regions(auc_neuron,fof_neuron,mpfc_neuron,2);
p_prior = kai2test_3regions(auc_neuron,fof_neuron,mpfc_neuron,3);

p_sound
p_sound(1)
p_sound(2)
p_choice
p_choice(3)
p_prior
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p = kai2test_3regions(auc_neuron,fof_neuron,mpfc_neuron,use_number)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
use_auc =  [auc_neuron(use_number), auc_neuron(4)- auc_neuron(use_number)];
use_fof =  [fof_neuron(use_number), fof_neuron(4)- fof_neuron(use_number)];
use_mpfc = [mpfc_neuron(use_number),mpfc_neuron(4)-mpfc_neuron(use_number)];
p(1) = kai2_test(use_auc,use_fof);
p(2) = kai2_test(use_auc,use_mpfc);
p(3) = kai2_test(use_mpfc,use_fof);
%p
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [p,median_sound] = ranksum_areas(auc_sound,fof_sound,mpfc_sound)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p(1)=ranksum(auc_sound,fof_sound);
p(2)=ranksum(auc_sound,mpfc_sound);
p(3)=ranksum(fof_sound,mpfc_sound);

length_auc = length(auc_sound);
length_fof = length(fof_sound);
length_mpfc = length(mpfc_sound);
length_auc = [1:length_auc] ./ length_auc;
length_fof = [1:length_fof] ./ length_fof;
length_mpfc = [1:length_mpfc] ./ length_mpfc;

hist_x = [-inf,-600:10:-400,inf];
hist_auc = histcounts(auc_sound,hist_x);
hist_fof = histcounts(fof_sound,hist_x);
hist_mpfc = histcounts(mpfc_sound,hist_x);

figure
subplot(2,1,1)
plot(sort(auc_sound),length_auc,'r')
hold on
plot(sort(fof_sound),length_fof,'g')
hold on
plot(sort(mpfc_sound),length_mpfc,'b')
subplot(2,1,2)
plot(hist_auc./sum(hist_auc),'r')
hold on
plot(hist_fof./sum(hist_fof),'g')
hold on
plot(hist_mpfc./sum(hist_mpfc),'b')
set(gca,'xlim',[0 length(hist_x)])
set(gca,'xtick',[1:2:length(hist_x)])
% boxplot([auc_sound;fof_sound;mpfc_sound],[ones(length(auc_sound),1);ones(length(fof_sound),1)*2;ones(length(mpfc_sound),1)*3])

median_sound = [median(auc_sound),median(fof_sound),median(mpfc_sound)];

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [BIC_sound,BIC_choice,BIC_prior,prob_sig,neuron_sig] = process_20231002_encoding_model2_depth(folders)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

analysis_dir = eval(folders);
analysis_dir

BIC_all = [];
LL_all = [];
y_sound_all = [];
y_choice_all = [];
y_prior_all = [];
for i = 1:30
    BIC_each(i).matrix = [];
    LL_each(i).matrix = [];
    
    y_sound_each(i).matrix = [];
    y_choice_each(i).matrix = [];
    y_prior_each(i).matrix = [];
end

for i = 1:length(analysis_dir)
    [i,length(analysis_dir)]
   
    [temp_BIC_each,temp_LL_each,temp_BIC_all,temp_LL_all, ...
        parameter_size,parameter_sound,parameter_choice,parameter_prior,...
        temp_sound_each,temp_choice_each,temp_prior_each, ...
        temp_sound_all,temp_choice_all,temp_prior_all] = ...
        Task_kaiseki_tokyo1_20231002_encoding_model_depth_control(analysis_dir{i});
    
    BIC_all = [BIC_all; temp_BIC_all];
    LL_all = [LL_all; temp_LL_all];
    y_sound_all = [y_sound_all; temp_sound_all];
    y_choice_all = [y_choice_all; temp_choice_all];
    y_prior_all = [y_prior_all; temp_prior_all];
    for j = 1:30
        BIC_each(j).matrix = [BIC_each(j).matrix; temp_BIC_each(j).matrix];
        LL_each(j).matrix = [LL_each(j).matrix; temp_LL_each(j).matrix];
        y_sound_each(j).matrix = [y_sound_each(j).matrix; temp_sound_each(j).matrix];
        y_choice_each(j).matrix = [y_choice_each(j).matrix; temp_choice_each(j).matrix];
        y_prior_each(j).matrix = [y_prior_each(j).matrix; temp_prior_each(j).matrix];        
    end
end
delete(gcp('nocreate'))

[BIC_sound,BIC_choice,BIC_prior,prob_sig,neuron_sig] = get_BIC_loglikeli_plot(BIC_all, LL_all,parameter_size,parameter_sound,parameter_choice,parameter_prior);
get_regress_plot(BIC_all, y_sound_all,y_choice_all,y_prior_all)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function get_regress_plot(BIC, y_sound,y_choice,y_prior)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%nan check
nanBIC = mean(BIC,2);
nanBIC = find(isnan(nanBIC) == 0);
BIC = BIC(nanBIC,:);
y_sound = y_sound(nanBIC,:);
y_choice = y_choice(nanBIC,:);
y_prior = y_prior(nanBIC,:);

%regression contains both the left and right side, need to split into half.
y_sound = get_value_regressor(y_sound);
y_choice = get_value_regressor(y_choice);
y_prior = get_value_regressor(y_prior);

y_sound = abs(y_sound);
y_choice = abs(y_choice);
y_prior = abs(y_prior);

figure
plot(mean(y_sound),'r')
hold on
plot(mean(y_choice),'g')
hold on
plot(mean(y_prior),'b')

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function use_y = get_value_regressor(y_sound)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[length_BIC,length_y] = size(y_sound);
for i = 1:length_BIC
    temp_y = y_sound(i,:);
    temp1 = temp_y(1:length_y/2);
    temp2 = temp_y(length_y/2 + 1:length_y);
    
    if max(temp1) >= max(temp2)
        use_y(i,:) = temp1;
    else
        use_y(i,:) = temp2;
    end
end
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [BIC_sound,BIC_choice,BIC_prior,prob_sig,neuron_sig] = get_BIC_loglikeli_plot(BIC, part_LL,parameter_size,parameter_sound,parameter_choice,parameter_prior)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%nan check
nanBIC = mean(BIC,2);
nanBIC = find(isnan(nanBIC) == 0);
BIC = BIC(nanBIC,:);
part_LL = part_LL(nanBIC,:);

BIC_sound = BIC(:,2) - BIC(:,1);
BIC_choice = BIC(:,3) - BIC(:,1);
BIC_prior = BIC(:,4) - BIC(:,1);

LL_sound = part_LL(:,1) - part_LL(:,2);
LL_choice = part_LL(:,1) - part_LL(:,3);
LL_prior = part_LL(:,1) - part_LL(:,4);

temp_x = [1:length(LL_sound)];

figure
subplot(2,2,1)
plot(sort(LL_sound),temp_x,'r')
hold on
plot(sort(LL_choice),temp_x,'g')
hold on
plot(sort(LL_prior),temp_x,'b')
subplot(2,2,2)
plot(sort(BIC_sound),temp_x,'r')
hold on
plot(sort(BIC_choice),temp_x,'g')
hold on
plot(sort(BIC_prior),temp_x,'b')
subplot(2,2,3)
plot(sort(LL_sound),'r')
hold on
plot(sort(LL_choice),'g')
hold on
plot(sort(LL_prior),'b')
subplot(2,2,4)
plot(sort(BIC_sound),'r')
hold on
plot(sort(BIC_choice),'g')
hold on
plot(sort(BIC_prior),'b')

p_LL_sound = 1-chi2cdf(2*LL_sound, parameter_size-parameter_sound);
p_LL_sound = -log10(p_LL_sound);
p_LL_choice = 1-chi2cdf(2*LL_choice, parameter_size-parameter_choice);
p_LL_choice = -log10(p_LL_choice);
p_LL_prior = 1-chi2cdf(2*LL_prior, parameter_size-parameter_prior);
p_LL_prior = -log10(p_LL_prior);

figure
plot(sort(p_LL_sound),'r')
hold on
plot(sort(p_LL_choice),'g')
hold on
plot(sort(p_LL_prior),'b')

sig_sound = find(p_LL_sound > 2);
sig_choice = find(p_LL_choice > 2);
sig_prior = find(p_LL_prior > 2);

sig_sc = intersect(sig_sound,sig_choice);
sig_cp = intersect(sig_prior,sig_choice);
sig_sp = intersect(sig_prior,sig_sound);
sig_scp = intersect(sig_sc,sig_prior);

length_scp = length(sig_scp);
length_sc = length(sig_sc) - length(sig_scp);
length_sp = length(sig_sp) - length(sig_scp);
length_cp = length(sig_cp) - length(sig_scp);
length_s = length(sig_sound)-length_sc-length_sp-length_scp;
length_c = length(sig_choice)-length_sc-length_cp-length_scp;
length_p = length(sig_prior)-length_sp-length_cp-length_scp;

[length_s, length_c, length_p]
[length_sc, length_sp, length_cp, length_scp]

temp_sig = [length(sig_sound),length(sig_choice),length(sig_prior)];
neuron_sig = [temp_sig,length(p_LL_sound)];
prob_sig = temp_sig ./ length(p_LL_sound)

return
