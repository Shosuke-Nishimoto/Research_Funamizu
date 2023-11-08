screens=Screen('Screens');
screenNumber=max(screens);

[expWin,rect]=Screen('OpenWindow',screenNumber);
% [expWin,rect]=Screen('OpenWindow',screenNumber,[],[1500 20 1900 300]);


%%
omega=12; % temporal period, in frames, of the drifting grating
frameRate=Screen('FrameRate',screenNumber);
stim_duration = 4;
numFrames = 2*frameRate*stim_duration;

x_0 = 15*1920/70.71; % 32インチの横幅70.71センチ分が1920
y_size = 1920; z_size = 1080;
[y,z] = meshgrid(-y_size/2:y_size/2,-z_size/2:z_size/2);

theta = acos(z./sqrt(x_0^2 + y.^2 + z.^2))-pi/2;
phi = atan(y/x_0);
target_theta = 0; target_phi = 10;
[z_center,y_center] = find(theta>(target_theta-0.08)*pi/180 & theta<(target_theta+0.08)*pi/180 ...
                        & phi>(target_phi-0.08)*pi/180 & phi<(target_phi+0.08)*pi/180);
z_center = z_center-z_size/2-1; y_center = y_center-y_size/2-1;

 %%
lambda = 32;  % 波長
sd     = 32; % グレーティングの大きさ
ori    = 45;
cnt    = 1;   % 刺激のコントラスト
lum = 0.5; % 平均輝度
f = 1;

for i=1:numFrames
    if i <= frameRate*stim_duration
        phase = (i/omega)*2*pi;
        % grating
        %input = exp(-((x-mean(mean(x))).^2+(y-mean(mean(y))).^2)/(2.*(sd.^2))); % 入力
        input = exp(-((y-y_center).^2+(z-z_center).^2)/(2.*(sd.^2))); % 入力
    
        gauss = input.*(1/max(abs(reshape(input,numel(input),1))));  % ガウスフィルタ
        grating = sin(2*pi.*(y.*cos(ori*pi/180)+z.*sin(ori*pi/180))/lambda+phase); % グレーティング
    
        im = gauss.*grating.*cnt;
        im = (1+im).*lum;
    else
        im = y*0+0.5;
    end
    
    im(1:50, end-50:end) = f;
    % 12Hz で切り替え
    if rem(i,5) == 0
        f = abs(f-1);
    end
    tex(i) = Screen('MakeTexture', expWin, im.*255);
end

%% 実行
movieDurationSecs=160;

movieDurationFrames=round(movieDurationSecs * frameRate);

movieFrameIndices=mod(0:(movieDurationFrames-1), numFrames) + 1;

for i=1:movieDurationFrames
    Screen('DrawTexture', expWin, tex(movieFrameIndices(i)));
    Screen('Flip', expWin);
end

%%

Screen('CloseAll')