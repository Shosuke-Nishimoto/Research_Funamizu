function [expWin, frameRate, numFrames, tex] = make_intrinsic_signal_imaging_visual_stimulus(x_0)
    screens=Screen('Screens');
    screenNumber=max(screens);
    [expWin,rect]=Screen('OpenWindow',screenNumber);
    %[expWin,rect]=Screen('OpenWindow',screenNumber,[],[1500 20 1900 300]);
    
    frameRate=Screen('FrameRate',screenNumber); % フレームレートを返す
    
    numFrames = 1300;
    
    %% A counter-phase checkerboard pattern was flashed on the bar, alternating between black and white (25° squares with 166 ms period)
    % The bar was 20° wide and subtended the whole visual hemifield along the vertical and horizontal axes (153° or 147° respectively). 
    % The bar drifted at 8.5-9.5°/s for intrinsic imaging experiments and at 12-14°/s for two-photon imaging experiments.
    
    fs = 25*pi/180; % 25° squares
    ft = 2; % 2 Hz temporal frequency
    %x_0 = 10*1920/70.71; % 32インチの横幅70.71センチ分が1920
    % [y,z] = meshgrid(-64:0.1:64, -36:0.1:36);
    [y,z] = meshgrid(-960:960, -540:540);
    theta = acos(z./sqrt(x_0^2 + y.^2 + z.^2))-pi/2;
    phi = atan(y/x_0);
%     theta = acos(y./sqrt(x_0^2 + y.^2 + z.^2))-pi/2;
%     phi = atan(-z/x_0);

    m_1 = make_spherically_corrected_checkerboard(fs,x_0);
    m_2 = abs(m_1-1);
    speed = 9/60*pi/180; % 1sで9度, 1 frame = 1/60s
    
    %%
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
        
        n(1:50, end-50:end) = 1;
        n = n.*255;
        tex(t)=Screen('MakeTexture', expWin, n);
        k = k+1;
        if rem(k,frameRate/6)==0 % 10 frames で切り替え
            hata = abs(hata-1);
        end
    end
    %%
    hata = 1;
    k = 0;
    angle = -pi/2;
    for t=numFrames+1:2*numFrames
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
        
        ind_saki = phi>=saki;
        ind_ato = phi<ato;
    %     ind_saki = theta>=saki;
    %     ind_ato = theta<ato;
        n(ind_saki) = 0.5;
        n(ind_ato) = 0.5;
        
        n(1:50, end-50:end) = 1;
        n = n.*255;
        tex(t)=Screen('MakeTexture', expWin, n);
        k = k+1;
        if rem(k,frameRate/6)==0 % 10 frames で切り替え
            hata = abs(hata-1);
        end
    end

end