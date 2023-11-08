x_0 = 10*1920/70.71; % 32インチの横幅70.71センチ分が1920
%x_0 = 10*1920/121.54; % 32インチの横幅121.54センチ分が1920

[expWin, frameRate, numFrames, tex] = make_intrinsic_signal_imaging_visual_stimulus(x_0);

%% 刺激提示

movieDurationSecs=600; % グレーティングがドリフトする時間
movieDurationFrames=round(movieDurationSecs * frameRate); % ドリフトしている時間全体のフレーム数
movieFrameIndices=mod(0:(movieDurationFrames-1), 2*numFrames) + 1;

for i=1:movieDurationFrames
    Screen('DrawTexture', expWin, tex(movieFrameIndices(i)));
    Screen('Flip', expWin);
end

Screen('CloseAll')