% 'Left_Go' or 'Right_Go'のフォルダー内で実行

folder = dir('S*');

Go_Target = 0; p = 0; % Go_Target trialの合計, そのうちGoを選択したtrialの合計
NoGo_Target = 0; q = 0;
Go_Control = 0; r = 0;
NoGo_Control = 0; s = 0;

for i = 1:length(folder)
    % フォルダーに移動
    cd(folder(i).name)
    list = dir('block*.mat');
    list = natsortfiles({list.name});
    NI_data = dir('NI*.mat');
    RawEvents = [];
    TrueSide = []; % left(Go):0, right(NoGo):1
    ChosenSide = []; % left(Go):0, NoChoice(NoGo):2
    Outcome = []; % incorrect(NoGoでGo):1, correct(GoでGo):2, NoChoice:3
    
    load(NI_data(1).name)
    for j = 2:length(list)
        load(list{i});
        RawEvents = [RawEvents saveBlock.RawEvents.Trial];
        TrueSide = [TrueSide saveBlock.TrueSide];
        ChosenSide = [ChosenSide saveBlock.ChosenSide];
        Outcome = [Outcome saveBlock.Outcome];
    end
    
    num = min(count, length(TrueSide));
    real_opt_schedule = opt_schedule(1:count,2)'; % Target:1, Control:2,3,...
    
    %%%
    % 終わりはそろっている
    real_opt_schedule = real_opt_schedule(end-num+1:end);
    RawEvents = RawEvents(end-num+1:end);
    TrueSide = TrueSide(end-num+1:end);
    ChosenSide = ChosenSide(end-num+1:end);
    Outcome = Outcome(end-num+1:end);
    %%%
    
    Go_Target = Go_Target + sum(TrueSide==0 & real_opt_schedule==1);
    NoGo_Target = NoGo_Target + sum(TrueSide==1 & real_opt_schedule==1);
    Go_Control = Go_Control + sum(TrueSide==0 & real_opt_schedule~=1);
    NoGo_Control = NoGo_Control + sum(TrueSide==1 & real_opt_schedule~=1);
    
    p = p + sum(ChosenSide(TrueSide==0 & real_opt_schedule==1)==0);
    % q = sum(ChosenSide(TrueSide==1 & real_opt_schedule==1)==2);
    q = q + sum(ChosenSide(TrueSide==1 & real_opt_schedule==1)==0);
    r = r + sum(ChosenSide(TrueSide==0 & real_opt_schedule~=1)==0);
    % s = sum(ChosenSide(TrueSide==1 & real_opt_schedule~=1)==2);
    s = s + sum(ChosenSide(TrueSide==1 & real_opt_schedule~=1)==0);
    
    % ディレクトリを一つ戻る
    cd ..
end

figure;
bar([p/Go_Target r/Go_Control; q/NoGo_Target s/NoGo_Control])
disp([p/Go_Target q/NoGo_Target r/Go_Control s/NoGo_Control])
disp([p Go_Target; q NoGo_Target; r Go_Control; s NoGo_Control])
lgd = legend('Target', 'Control');
lgd.FontSize = 18;

% カイ二乗検定
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