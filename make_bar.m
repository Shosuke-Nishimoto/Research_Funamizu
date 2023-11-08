data_go = [0.85
1
0.9896
1
0.9885
0.9883
1
0.954
0.929
0.9733
0.871
1
1];
data_go_av = mean(data_go);
data_go_std = std(data_go);

data_nogo = [0.303
0.4314
0.4626
0.2932
0.4491
0.1769
0.4828
0.4559
0.4486
0.3611
0.28
0.5741
0.5412];
data_nogo_av = mean(data_nogo);
data_nogo_std = std(data_nogo);

%%
data_av = [data_go_av, data_nogo_av];
data_sem = [data_go_std, data_nogo_std]./sqrt(length(data_go));
ts = tinv([0.025  0.975],length(data_go)-1);
CI = data_sem.*ts;
x = 1:2;               

bar(x, data_av)
hold on;
errorbar(x,data_av,CI, "black", "LineStyle","none")
ylim([0 1])

hold off;

%% t test

data = data_go - data_nogo;
[h,p,ci,stats] = ttest(data);

