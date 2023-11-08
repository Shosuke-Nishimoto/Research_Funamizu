screens=Screen('Screens');
screenNumber=max(screens);
%[expWin,rect]=Screen('OpenWindow',screenNumber);
[expWin,rect]=Screen('OpenWindow',screenNumber,[],[1500 20 1900 300]);

movieDurationSecs=25; % グレーティングがドリフトする時間
frameRate=Screen('FrameRate',screenNumber); % フレームレートを返す
movieDurationFrames=round(movieDurationSecs * frameRate); % ドリフトしている時間全体のフレーム数

numFrames = 2000;
movieFrameIndices=mod(0:(movieDurationFrames-1), numFrames) + 1;
%% A counter-phase checkerboard pattern was flashed on the bar, alternating between black and white (25° squares with 166 ms period)
% The bar was 20° wide and subtended the whole visual hemifield along the vertical and horizontal axes (153° or 147° respectively). 
% The bar drifted at 8.5-9.5°/s for intrinsic imaging experiments and at 12-14°/s for two-photon imaging experiments.

fs = 180/25/pi; % 25° squares
ft = 2; % 2 Hz temporal frequency
x_0 = 1;
[y,z] = meshgrid(-10:0.04:10, -10:0.04:10);
theta = acos(z./sqrt(x_0^2 + y.^2 + z.^2))-pi/2;
phi = atan(y/x_0);

m_1 = make_spherically_corrected_checkerboard(fs,x_0);
m_2 = abs(m_1-1);
speed = 9/60*pi/180; % 1sで9度, 1 frame = 1/60s
hata = 1;
k = 0;
angle = -pi/2;

for t=1:numFrames
    %n = m;
    if hata==1
        n = m_1;
    else
        n = m_2;
    end

    angle = angle+speed;
%     disp(angle)
    saki = min(angle,pi/2);
    ato = max(-pi/2,angle-20*pi/180);
    
%     ind_saki = phi>=saki;
%     ind_ato = phi<ato;
    ind_saki = theta>=saki;
    ind_ato = theta<ato;
    n(ind_saki) = 0.5;
    n(ind_ato) = 0.5;
    
    n = n.*255;
    tex(t)=Screen('MakeTexture', expWin, n); %#ok<AGROW>
    k = k+1;
    if rem(k,frameRate/6)==0 % 10 frames で切り替え
        hata = abs(hata-1);
    end
end

%%
for i=1:movieDurationFrames
    Screen('DrawTexture', expWin, tex(movieFrameIndices(i)));
    Screen('Flip', expWin);
end

%%
Screen('CloseAll')
%%
function image = make_spherically_corrected_checkerboard(fs,x_0)
    [y,z] = meshgrid(-10:0.04:10, -10:0.04:10);
    % alpha rotation
    % alpha = pi/2;
    % y_2 = z.*sin(-alpha) + y.*cos(-alpha);
    % z_2 = z.*cos(-alpha) - y.*sin(-alpha);

    theta = pi/2-acos(z./sqrt(x_0^2 + y.^2 + z.^2));
    phi = atan(-y/x_0);
    s_1 = cos(2*pi*fs.*theta);
    s_2 = cos(2*pi*fs.*phi);

    s_1 = abs(s_1);
    s_2 = abs(s_2);
    
    % 25 degree きざみ
    ind_small_1 = (abs(theta)>=0 & abs(theta)<12.5/180*pi) | (abs(theta)>=37.5/180*pi & abs(theta)<62.5/180*pi)...
                                | (abs(theta)>=87.5/180*pi);
    ind_large_1 = (abs(theta)>=12.5/180*pi & abs(theta)<37.5/180*pi) | (abs(theta)>=62.5/180*pi & abs(theta)<87.5/180*pi);
    ind_small_2 = (abs(phi)>=0 & abs(phi)<12.5/180*pi) | (abs(phi)>=37.5/180*pi & abs(phi)<62.5/180*pi)...
                                | (abs(phi)>=87.5/180*pi);
    ind_large_2 = (abs(phi)>=12.5/180*pi & abs(phi)<37.5/180*pi) | (abs(phi)>=62.5/180*pi & abs(phi)<87.5/180*pi);
    
    s_1(ind_small_1) = 0;
    s_1(ind_large_1) = 1;
    s_2(ind_small_2) = 0;
    s_2(ind_large_2) = -1;
    
    image = abs(s_1 + s_2);
end