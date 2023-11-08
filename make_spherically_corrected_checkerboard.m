function image = make_spherically_corrected_checkerboard(fs,x_0)
%     [y,z] = meshgrid(-64:0.1:64, -36:0.1:36);
    [y,z] = meshgrid(-960:960, -540:540);
    % alpha rotation
    % alpha = pi/2;
    % y_2 = z.*sin(-alpha) + y.*cos(-alpha);
    % z_2 = z.*cos(-alpha) - y.*sin(-alpha);

    theta = pi/2-acos(z./sqrt(x_0^2 + y.^2 + z.^2));
    phi = atan(-y/x_0);
%     theta = pi/2-acos(y./sqrt(x_0^2 + y.^2 + z.^2));
%     phi = atan(-z/x_0);

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