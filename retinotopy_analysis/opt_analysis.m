list = dir('block*.mat');
list = natsortfiles({list.name});
RawEvents = [];
TrueSide = []; % left(Go):0, right(NoGo):1
ChosenSide = []; % left(Go):0, NoChoice(NoGo):2
Outcome = []; % incorrect(NoGoÇ≈Go):1, correct(GoÇ≈Go):2, NoChoice:3
Opto_trial = []; % Photo_stimulus:1, No_Photo:0

for i = 2:length(list)
    load(list{i});
    RawEvents = [RawEvents saveBlock.RawEvents.Trial];
    TrueSide = [TrueSide saveBlock.TrueSide];
    ChosenSide = [ChosenSide saveBlock.ChosenSide];
    Outcome = [Outcome saveBlock.Outcome];
    Opto_trial = [Opto_trial saveBlock.Opto_trial];
end

Go_Target = sum(TrueSide==0 & Opto_trial==1);
NoGo_Target = sum(TrueSide==1 & Opto_trial==1);
Go_Control = sum(TrueSide==0 & Opto_trial~=1);
NoGo_Control = sum(TrueSide==1 & Opto_trial~=1);

p = sum(ChosenSide(TrueSide==0 & Opto_trial==1)==0);
% q = sum(ChosenSide(TrueSide==1 & real_opt_schedule==1)==2);
q = sum(ChosenSide(TrueSide==1 & Opto_trial==1)==0);
r = sum(ChosenSide(TrueSide==0 & Opto_trial~=1)==0);
% s = sum(ChosenSide(TrueSide==1 & real_opt_schedule~=1)==2);
s = sum(ChosenSide(TrueSide==1 & Opto_trial~=1)==0);

figure;
bar([p/Go_Target q/NoGo_Target r/Go_Control s/NoGo_Control])
disp([p/Go_Target q/NoGo_Target r/Go_Control s/NoGo_Control])
disp([p Go_Target; q NoGo_Target; r Go_Control; s NoGo_Control])

% ÉJÉCìÒèÊåüíË
kai2_test([p, Go_Target-p], [r, Go_Control-r]);
kai2_test([q, NoGo_Target-q], [s, NoGo_Control-s]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p = kai2_test(sample1,sample2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%sample: [data1, data2]

sum1 = sum(sample1);
sum2 = sum(sample2);

sum_data1 = sample1(1) + sample2(1);
sum_data2 = sample1(2) + sample2(2);

sum_all = sum1 + sum2

expect1 = sum1 ./ sum_all;
expect2 = sum2 ./ sum_all;

expect11 = expect1 * sum_data1;
expect12 = expect1 * sum_data2;
expect21 = expect2 * sum_data1;
expect22 = expect2 * sum_data2;

dif11 = ((sample1(1) - expect11)^2) / expect11;
dif12 = ((sample1(2) - expect12)^2) / expect12;
dif21 = ((sample2(1) - expect21)^2) / expect21;
dif22 = ((sample2(2) - expect22)^2) / expect22;

dif_all = dif11 + dif12 + dif21 + dif22;

p = chi2cdf(dif_all,1,'upper')

end