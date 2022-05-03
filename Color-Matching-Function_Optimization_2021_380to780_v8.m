clear all;
clc;
format long;
options = struct('MaxFunEvals',10000);
weight_pl = 0.9;
weight_sl = 0.1;

fprintf('Progress: 0%%\n');
%% Read data and interpolate
wl = xlsread('1000CMFs.xlsx', 'Wavelengths');
wl = wl(1:401,:);
Xb = xlsread('1000CMFs.xlsx', 'Xb_2015');
Xb = Xb(1:401,:);
Yb = xlsread('1000CMFs.xlsx', 'Yb_2015');
Yb = Yb(1:401,:);
Zb = xlsread('1000CMFs.xlsx', 'Zb_2015');
Zb = Zb(1:401,:);
opt_data_ph1 = xlsread('Optimization_SPDs.xlsx', 'Phase 1');
opt_data_ph2 = xlsread('Optimization_SPDs.xlsx', 'Phase 2');
ci = [4,0,0,0,9,0,1,15,3];

spds_ph1 = zeros(length(wl)+1,size(opt_data_ph1,2));
spds_ph1(1,:) = opt_data_ph1(1,:);
spds_ph1(2:end,1) = wl;
for i = 2:size(opt_data_ph1,2)
    spds_ph1(2:end,i) = interp1(opt_data_ph1(2:end,1), opt_data_ph1(2:end,i), wl);
end
spds_ph1(isnan(spds_ph1)) = 0;

spds_ph2 = zeros(length(wl)+1,size(opt_data_ph2,2));
spds_ph2(1,:) = opt_data_ph2(1,:);
spds_ph2(2:end,1) = wl;
for i = 2:size(opt_data_ph2,2)
    spds_ph2(2:end,i) = interp1(opt_data_ph2(2:end,1), opt_data_ph2(2:end,i), wl);
end
spds_ph2(isnan(spds_ph2)) = 0;

%% Phase 1 optimization
% Calculate original XYZ and st for Phase 1 data %
% X1 = zeros(size(Xb,2),size(spds_ph1,2)-1);
% Y1 = zeros(size(Xb,2),size(spds_ph1,2)-1);
% Z1 = zeros(size(Xb,2),size(spds_ph1,2)-1);
% s_ori1 = zeros(size(Xb,2),size(spds_ph1,2)-1);
% t_ori1 = zeros(size(Xb,2),size(spds_ph1,2)-1);
% for i = 1:size(Xb,2)
%     for j = 1:size(spds_ph1,2)-1
%         X1(i,j) = Xb(:,i)'*spds_ph1(2:end,j+1);
%         Y1(i,j) = Yb(:,i)'*spds_ph1(2:end,j+1);
%         Z1(i,j) = Zb(:,i)'*spds_ph1(2:end,j+1);
%         s_ori1(i,j) = 4*X1(i,j) / (X1(i,j) + 15*Y1(i,j) + 3*Z1(i,j));
%         t_ori1(i,j) = 9*Y1(i,j) / (X1(i,j) + 15*Y1(i,j) + 3*Z1(i,j));
%     end
% end
% 
% % Baseline (bl) of Spectrum Loci (sl) %
% baseline1 = xlsread('SpectrumLoci_baseline.xlsx');
% wl1 = [400,455,468,475,480,485,491,498,512,540,561,576,587,598,610,627,780]';
% index = [];
% for i = 1:length(wl1)
%     index(i) = find(baseline1(:,1) == wl1(i));
% end
% s_sl_bl1 = baseline1(index,2);
% t_sl_bl1 = baseline1(index,3);
% 
% % Optimization %
% coeff_mean1 = zeros(size(Xb,2),length(ci));
% coeff_rms1 = zeros(size(Xb,2),length(ci));
% for i=1:size(Xb,2)
%     fun_mean = @(cm1)mean(sqrt(((cm1(1)*X1(i,:)+cm1(2)*Y1(i,:)+cm1(3)*Z1(i,:))/(cm1(7)*X1(i,:)+cm1(8)*Y1(i,:)+cm1(9)*Z1(i,:))-s_sl_bl1).^2 + ...
%     ((cm1(4)*X1(i,:)+cm1(5)*Y1(i,:)+cm1(6)*Z1(i,:))/(cm1(7)*X1(i,:)+cm1(8)*Y1(i,:)+cm1(9)*Z1(i,:))-t_sl_bl1).^2));
%     fun_rms = @(cr1)rms(sqrt(((cr1(1)*X1(i,:)+cr1(2)*Y1(i,:)+cr1(3)*Z1(i,:))/(cr1(7)*X1(i,:)+cr1(8)*Y1(i,:)+cr1(9)*Z1(i,:))-s_sl_bl1).^2 + ...
%     ((cr1(4)*X1(i,:)+cr1(5)*Y1(i,:)+cr1(6)*Z1(i,:))/(cr1(7)*X1(i,:)+cr1(8)*Y1(i,:)+cr1(9)*Z1(i,:))-t_sl_bl1).^2));
%     
%     [cm1,fmval] = fminunc(fun_mean,ci,options);
%     [cr1,frval] = fminunc(fun_rms,ci,options);
% 
%     coeff_mean1(i,:) = cm1;
%     coeff_rms1(i,:) = cr1;
% end

%% Phase 2 Optimization
% pl_bl2 = xlsread('PlankianLocus_baseline.xlsx');
% sl_bl2 = xlsread('SpectrumLoci_baseline.xlsx');
% sl_bl2(isnan(sl_bl2)) = 0;
% cct2 = spds_ph2(1,2:7);
% wl2 = spds_ph2(1,8:end);
% 
% index = [];
% for i = 1:length(wl2)
%     index(i) = find(sl_bl2(:,1) == wl2(i));
% end
% s_sl_bl2 = sl_bl2(index,2);
% t_sl_bl2 = sl_bl2(index,3);
% 
% index = [];
% for i = 1:length(cct2)
%     index(i) = find(pl_bl2(:,1) == cct2(i));
% end
% s_pl_bl2 = pl_bl2(index,2);
% t_pl_bl2 = pl_bl2(index,3);
% 
% s_bl2 = [s_pl_bl2' s_sl_bl2'];
% t_bl2 = [t_pl_bl2' t_sl_bl2'];
% 
% X2 = zeros(size(Xb,2),size(spds_ph2,2)-1);
% Y2 = zeros(size(Xb,2),size(spds_ph2,2)-1);
% Z2 = zeros(size(Xb,2),size(spds_ph2,2)-1);
% s_ori2 = zeros(size(Xb,2),size(spds_ph2,2)-1);
% t_ori2 = zeros(size(Xb,2),size(spds_ph2,2)-1);
% for i = 1:size(Xb,2)
%     for j = 1:size(spds_ph2,2)-1
%         X2(i,j) = Xb(:,i)'*spds_ph2(2:end,j+1);
%         Y2(i,j) = Yb(:,i)'*spds_ph2(2:end,j+1);
%         Z2(i,j) = Zb(:,i)'*spds_ph2(2:end,j+1);
%         s_ori2(i,j) = 4*X2(i,j) / (X2(i,j) + 15*Y2(i,j) + 3*Z2(i,j));
%         t_ori2(i,j) = 9*Y2(i,j) / (X2(i,j) + 15*Y2(i,j) + 3*Z2(i,j));
%     end
% end
% 
% coeff_mean2 = zeros(size(Xb,2),length(ci));
% coeff_rms2 = zeros(size(Xb,2),length(ci));
% for i = 1:size(Xb,2)
%     fun_mean2 = @(cm2)(mean(weight_pl*sqrt(((cm2(1)*X2(i,1:6)+cm2(2)*Y2(i,1:6)+cm2(3)*Z2(i,1:6))./(cm2(7)*X2(i,1:6)+cm2(8)*Y2(i,1:6)+cm2(9)*Z2(i,1:6))-s_pl_bl2').^2 + ...
%     ((cm2(4)*X2(i,1:6)+cm2(5)*Y2(i,1:6)+cm2(6)*Z2(i,1:6))./(cm2(7)*X2(i,1:6)+cm2(8)*Y2(i,1:6)+cm2(9)*Z2(i,1:6))-t_pl_bl2').^2)) + ...
%     mean(weight_sl*sqrt(((cm2(1)*X2(i,7:end)+cm2(2)*Y2(i,7:end)+cm2(3)*Z2(i,7:end))./(cm2(7)*X2(i,7:end)+cm2(8)*Y2(i,7:end)+cm2(9)*Z2(i,7:end))-s_sl_bl2').^2 + ...
%     ((cm2(4)*X2(i,7:end)+cm2(5)*Y2(i,7:end)+cm2(6)*Z2(i,7:end))./(cm2(7)*X2(i,7:end)+cm2(8)*Y2(i,7:end)+cm2(9)*Z2(i,7:end))-t_sl_bl2').^2)));
% 
%     fun_rms2 = @(cr2)(rms(weight_pl*sqrt(((cr2(1)*X2(i,1:6)+cr2(2)*Y2(i,1:6)+cr2(3)*Z2(i,1:6))./(cr2(7)*X2(i,1:6)+cr2(8)*Y2(i,1:6)+cr2(9)*Z2(i,1:6))-s_pl_bl2').^2 + ...
%     ((cr2(4)*X2(i,1:6)+cr2(5)*Y2(i,1:6)+cr2(6)*Z2(i,1:6))./(cr2(7)*X2(i,1:6)+cr2(8)*Y2(i,1:6)+cr2(9)*Z2(i,1:6))-t_pl_bl2').^2)) + ...
%     rms(weight_sl*sqrt(((cr2(1)*X2(i,7:end)+cr2(2)*Y2(i,7:end)+cr2(3)*Z2(i,7:end))./(cr2(7)*X2(i,7:end)+cr2(8)*Y2(i,7:end)+cr2(9)*Z2(i,7:end))-s_sl_bl2').^2 + ...
%     ((cr2(4)*X2(i,7:end)+cr2(5)*Y2(i,7:end)+cr2(6)*Z2(i,7:end))./(cr2(7)*X2(i,7:end)+cr2(8)*Y2(i,7:end)+cr2(9)*Z2(i,7:end))-t_sl_bl2').^2)));
%     
%     [cm2,fmval2] = fminsearch(fun_mean2,coeff_mean1(i,:),options);
%     [cr2,frval2] = fminsearch(fun_rms2,coeff_rms1(i,:),options);
% 
%     coeff_mean2(i,:) = cm2;
%     coeff_rms2(i,:) = cr2;
% end
% 
% s_ph2_optmean = zeros(size(Xb,2),size(spds_ph2,2)-1);
% t_ph2_optmean = zeros(size(Xb,2),size(spds_ph2,2)-1);
% s_ph2_optrms = zeros(size(Xb,2),size(spds_ph2,2)-1);
% t_ph2_optrms = zeros(size(Xb,2),size(spds_ph2,2)-1);
% for i = 1:size(Xb,2)
%     for j = 1:size(spds_ph2,2)-1
%         s_ph2_optmean(i,j) = (coeff_mean2(i,1:3)*[X2(i,j),Y2(i,j),Z2(i,j)]') / (coeff_mean2(i,7:9)*[X2(i,j),Y2(i,j),Z2(i,j)]');
%         t_ph2_optmean(i,j) = (coeff_mean2(i,4:6)*[X2(i,j),Y2(i,j),Z2(i,j)]') / (coeff_mean2(i,7:9)*[X2(i,j),Y2(i,j),Z2(i,j)]');
%         s_ph2_optrms(i,j) = (coeff_rms2(i,1:3)*[X2(i,j),Y2(i,j),Z2(i,j)]') / (coeff_rms2(i,7:9)*[X2(i,j),Y2(i,j),Z2(i,j)]');
%         t_ph2_optrms(i,j) = (coeff_rms2(i,4:6)*[X2(i,j),Y2(i,j),Z2(i,j)]') / (coeff_rms2(i,7:9)*[X2(i,j),Y2(i,j),Z2(i,j)]');
%     end
% end

%% Data analysis
% dst_ori = zeros(size(Xb,2),size(spds_ph2,2)-1);
% dst_mean = zeros(size(Xb,2),size(spds_ph2,2)-1);
% dst_rms = zeros(size(Xb,2),size(spds_ph2,2)-1);
% for i = 1:size(Xb,2)
%    for j = 1: size(spds_ph2,2)-1
%        dst_ori(i,j) = sqrt((s_ori2(i,j)-s_bl2(j))^2+(t_ori2(i,j)-t_bl2(j))^2);
%        dst_mean(i,j) = sqrt((s_ph2_optmean(i,j)-s_bl2(j))^2+(t_ph2_optmean(i,j)-t_bl2(j))^2);
%        dst_rms(i,j) = sqrt((s_ph2_optrms(i,j)-s_bl2(j))^2+(t_ph2_optrms(i,j)-t_bl2(j))^2);
%    end
% end
% dst_avg = zeros(size(Xb,2),4);
% for i = 1:size(Xb,2)
%    dst_avg(i,1) = (weight_pl*sum(dst_ori(i,1:6))+weight_sl*sum(dst_ori(i,7:end)))/(size(spds_ph2,2)-1);
%    dst_avg(i,2) = (weight_pl*sum(dst_mean(i,1:6))+weight_sl*sum(dst_mean(i,7:end)))/(size(spds_ph2,2)-1);
%    dst_avg(i,3) = (weight_pl*sum(dst_rms(i,1:6))+weight_sl*sum(dst_rms(i,7:end)))/(size(spds_ph2,2)-1);
%    dst_avg(i,4) = find(dst_avg(i,1:3)==min(dst_avg(i,1:3)));
% end

fprintf('Progress: 25%%\n');
coeff_best = xlsread('Best_Coeffs.xlsx');
% coeff_best = zeros(size(Xb,2),length(ci));
% for i = 1:size(Xb,2)
%     if dst_avg(i,4) == 1
%         coeff_best(i,:) = ci;
%     elseif dst_avg(i,4) == 2
%         coeff_best(i,:) = coeff_mean2(i,:);
%     else
%         coeff_best(i,:) = coeff_rms2(i,:);
%     end
% end

% s_optbest = zeros(size(Xb,2),size(spds_ph2,2)-1);
% t_optbest = zeros(size(Xb,2),size(spds_ph2,2)-1);
% for i = 1:size(Xb,2)
%     for j = 1:size(spds_ph2,2)-1
%         s_optbest(i,j) = (coeff_best(i,1:3)*[X2(i,j),Y2(i,j),Z2(i,j)]') / (coeff_best(i,7:9)*[X2(i,j),Y2(i,j),Z2(i,j)]');
%         t_optbest(i,j) = (coeff_best(i,4:6)*[X2(i,j),Y2(i,j),Z2(i,j)]') / (coeff_best(i,7:9)*[X2(i,j),Y2(i,j),Z2(i,j)]');
%     end
% end

%% Calculate Reference SPDs
% DSeries = xlsread('DSeries.xlsx');
% S0 = DSeries(:,2);
% S1 = DSeries(:,3);
% S2 = DSeries(:,4);
% CT_temps = xlsread('CT_temps.xlsx');
% CT = CT_temps(:,1);
% 
% Sr = zeros(length(wl),length(CT));
% for j = 1:length(CT)
%     if CT(j) <= 7000
%        xD = -4.6070E9/CT(j)^3 + 2.9678E6/CT(j)^2 + 0.09911E3/CT(j) + 0.244063;
%     else
%        xD = -2.0064E9/CT(j)^3 + 1.9018E6/CT(j)^2 + 0.24748E3/CT(j) + 0.23704;
%     end
%     yD = -3.000*xD^2 + 2.870*xD - 0.275;
%     M1 = (-1.3515 - 1.7703*xD + 5.9114*yD) / (0.0241 + 0.2562*xD - 0.7341*yD);
%     M2 = (0.0300 - 31.4424*xD + 30.0717*yD) / (0.0241 + 0.2562*xD - 0.7341*yD);
%     
%     S_rP = zeros(length(wl),1);
%     for i = 1:length(wl)
%         S_rP(i) = (wl(i)/1E9)^(-5)*(exp(1.4388E-2/(wl(i)/1E9)/CT(j))-1)^(-1) / ((560/1E9)^(-5)*(exp(1.4388E-2/(560/1E9)/CT(j))-1)^(-1));
%     end
%     S_rD = S0 + M1*S1 + M2*S2;
%     S_rM = (5000-CT(j))/1000*S_rP + (1-(5000-CT(j))/1000)*S_rD;
%     
%     if CT(j) <= 4000
%         Sr(:,j) = S_rP;
%     elseif CT(j) >= 5000
%         Sr(:,j) = S_rD;
%     else
%         Sr(:,j) = S_rM;
%     end
%     Sr(:,j) = Sr(:,j)/max(Sr(:,j));
% end

fprintf('Progress: 50%%\n');
%% Calculate Planckian SPDs
CT_temps = xlsread('CT_temps.xlsx');
CT = CT_temps(:,1);

Sr = zeros(length(wl),length(CT));
for j = 1:length(CT)
    S_rP = zeros(length(wl),1);
    for i = 1:length(wl)
        S_rP(i) = (wl(i)/1E9)^(-5)*(exp(1.4388E-2/(wl(i)/1E9)/CT(j))-1)^(-1) / ((560/1E9)^(-5)*(exp(1.4388E-2/(560/1E9)/CT(j))-1)^(-1));
    end
    Sr(:,j) = S_rP;
    Sr(:,j) = Sr(:,j)/max(Sr(:,j));
end

%% Calculate CT (s,t) baselines with Original (ori) and Optimized (opt) transforms
X_ct = zeros(size(Xb,2),length(CT));
Y_ct = zeros(size(Xb,2),length(CT));
Z_ct = zeros(size(Xb,2),length(CT));
s_ct_bl_ori = zeros(size(Xb,2),length(CT));
t_ct_bl_ori = zeros(size(Xb,2),length(CT));
s_ct_bl_opt = zeros(size(Xb,2),length(CT));
t_ct_bl_opt = zeros(size(Xb,2),length(CT));
for j = 1:length(CT)
    for i = 1:size(Xb,2)
        X_ct(i,j) = Xb(:,i)'*Sr(:,j);
        Y_ct(i,j) = Yb(:,i)'*Sr(:,j);
        Z_ct(i,j) = Zb(:,i)'*Sr(:,j);
        s_ct_bl_ori(i,j) = (ci(1:3)*[X_ct(i,j),Y_ct(i,j),Z_ct(i,j)]') / (ci(7:9)*[X_ct(i,j),Y_ct(i,j),Z_ct(i,j)]');
        t_ct_bl_ori(i,j) = (ci(4:6)*[X_ct(i,j),Y_ct(i,j),Z_ct(i,j)]') / (ci(7:9)*[X_ct(i,j),Y_ct(i,j),Z_ct(i,j)]');
        s_ct_bl_opt(i,j) = (coeff_best(i,1:3)*[X_ct(i,j),Y_ct(i,j),Z_ct(i,j)]') / (coeff_best(i,7:9)*[X_ct(i,j),Y_ct(i,j),Z_ct(i,j)]');
        t_ct_bl_opt(i,j) = (coeff_best(i,4:6)*[X_ct(i,j),Y_ct(i,j),Z_ct(i,j)]') / (coeff_best(i,7:9)*[X_ct(i,j),Y_ct(i,j),Z_ct(i,j)]');
    end
end

%% Calculate (s,t) coordinates of 1528 Sample SPDs
SPD_samples = xlsread('SampleSPDs.xlsx');
X_samples = zeros(size(Xb,2),size(SPD_samples,2));
Y_samples = zeros(size(Xb,2),size(SPD_samples,2));
Z_samples = zeros(size(Xb,2),size(SPD_samples,2));
s_samples = zeros(size(Xb,2),size(SPD_samples,2));
t_samples = zeros(size(Xb,2),size(SPD_samples,2));
for j = 1:size(SPD_samples,2)
    SPDj = (SPD_samples(4:end,j) + abs(SPD_samples(4:end,j))) / 2;
    SPDj(isnan(SPDj)) = [];
    for i = 1:size(Xb,2)
        Xb_interp = interp1(wl, Xb(:,i), (SPD_samples(1,j):SPD_samples(3,j):SPD_samples(2,j)));
        Yb_interp = interp1(wl, Yb(:,i), (SPD_samples(1,j):SPD_samples(3,j):SPD_samples(2,j)));
        Zb_interp = interp1(wl, Zb(:,i), (SPD_samples(1,j):SPD_samples(3,j):SPD_samples(2,j)));
        Xb_interp(isnan(Xb_interp)) = 0;
        Yb_interp(isnan(Yb_interp)) = 0;
        Zb_interp(isnan(Zb_interp)) = 0;
        X_samples(i,j) = Xb_interp * SPDj;
        Y_samples(i,j) = Yb_interp * SPDj;
        Z_samples(i,j) = Zb_interp * SPDj;
        s_samples(i,j) = (coeff_best(i,1:3)*[X_samples(i,j),Y_samples(i,j),Z_samples(i,j)]') / (coeff_best(i,7:9)*[X_samples(i,j),Y_samples(i,j),Z_samples(i,j)]');
        t_samples(i,j) = (coeff_best(i,4:6)*[X_samples(i,j),Y_samples(i,j),Z_samples(i,j)]') / (coeff_best(i,7:9)*[X_samples(i,j),Y_samples(i,j),Z_samples(i,j)]');
    end
end

%% Calculate CCTst and Dst
CCTst_samples = zeros(size(Xb,2),size(SPD_samples,2));
Dst_samples = zeros(size(Xb,2),size(SPD_samples,2));
for i = 1:size(Xb,2)
    for j = 1:size(SPD_samples,2)
        for m = 1:2
            d_in(m) = sqrt((s_samples(i,j) - s_ct_bl_opt(i,m))^2 + (t_samples(i,j) - t_ct_bl_opt(i,m))^2);
        end
        for m = 3:length(CT)
            d_in(m) = sqrt((s_samples(i,j) - s_ct_bl_opt(i,m))^2 + (t_samples(i,j) - t_ct_bl_opt(i,m))^2);
            if d_in(m-2) > d_in(m-1) && d_in(m-1) < d_in(m)
                break
            end
        end
 
        m = m - 1;
        
        if m < 2 | m > length(CT)-1
            Calc_CCTst = 1;
            Calc_Dst = 1;
            break
        end
        
        for k = 1:3
            cctst(k) = CT(k+m-2);
            s(k) = s_ct_bl_opt(i,k+m-2);
            t(k) = t_ct_bl_opt(i,k+m-2);
            d(k) = sqrt((s_samples(i,j) - s_ct_bl_opt(i,k+m-2))^2 + (t_samples(i,j) - t_ct_bl_opt(i,k+m-2))^2);
        end
        
        %Triangular Solution
        l_tri = sqrt((s(3)-s(1))^2 + (t(3)-t(1))^2);
        x = (d(1)^2 - d(3)^2 + l_tri^2) / (2 * l_tri);
        CCTst_triangular = cctst(1) + (cctst(3)-cctst(1))*x / l_tri;
        Dst_triangular_dist = sqrt(d(1)^2 - x^2);
        
        %Dst Sign
        t_Tx = t(1) + (t(3)-t(1))*x / l_tri;
        if t_samples(i,j) > t_Tx
           sign = 1;
        else
           sign = -1;
        end
        
        %Triangular Solution Dst
        Dst_triangular = Dst_triangular_dist * sign;
        
        %Parabolic Solution
        a_ = d(1) / (cctst(1) - cctst(2)) / (cctst(1) - cctst(3));
        b_ = d(2) / (cctst(2) - cctst(1)) / (cctst(2) - cctst(3));
        c_ = d(3) / (cctst(3) - cctst(1)) / (cctst(3) - cctst(2));
        A = a_ + b_ + c_;
        B = -1 * (a_ * (cctst(3) + cctst(2)) + b_ * (cctst(1) + cctst(3)) + c_ * (cctst(2) + cctst(1)));
        c = a_ * cctst(2) * cctst(3) + b_ * cctst(1) * cctst(3) + c_ * cctst(1) * cctst(2);
        CCTst_parabolic = -B / (2 * A);
        Dst_parabolic = (A * CCTst_parabolic^2 + B * CCTst_parabolic + c) * sign;
        
        %Shifted Triangular Solution
        CCTst_triangular_shift = CCTst_triangular + (CCTst_parabolic - CCTst_triangular) * Dst_triangular_dist * (1 / 0.002);
        
        %Set final CCT
        if Dst_triangular_dist < 0.002
           Calc_CCTst = CCTst_triangular_shift;
        else
           Calc_CCTst = CCTst_parabolic;
        end
        
        %Set final Dst
        if Dst_triangular_dist < 0.002
           Calc_Dst = Dst_triangular;
        else
           Calc_Dst = Dst_parabolic;
        end
        
        CCTst_samples(i,j) = Calc_CCTst;
        Dst_samples(i,j) = Calc_Dst;
    end
end

fprintf('Progress: 75%%\n');
%% Cross Check with MR's results
s_mr = xlsread('Data check\data_mr.xlsx','s');
t_mr = xlsread('Data check\data_mr.xlsx','t');
CCTst_mr = xlsread('Data check\data_mr.xlsx','CCTst');
Dst_mr = xlsread('Data check\data_mr.xlsx','Dst');
SPDs_mr = xlsread('Data check\data_mr.xlsx','SPDs');

figure(101)
plot(s_samples, t_samples, 'b.'); hold on; plot(s_mr, t_mr, 'r.'); hold off;
figure(102)
plot(s_mr, t_mr, 'r.'); hold on; plot(s_samples, t_samples, 'b.'); hold off;

figure(103)
plot(CCTst_samples, Dst_samples, 'b.'); hold on; plot(CCTst_mr, Dst_mr, 'r.'); hold off;
figure(104)
plot(CCTst_mr, Dst_mr, 'r.'); hold on; plot(CCTst_samples, Dst_samples, 'b.'); hold off;

%% Plot
% prompt1 = 'Input a CMF# (1~1000): ';
% CMF = input(prompt1);
% figure(1)
% hold on;
% plot([s_ori2(CMF,1:6),NaN,s_ori2(CMF,7:end)],[t_ori2(CMF,1:6),NaN,t_ori2(CMF,7:end)],'b','LineWidth',2);
% plot([s_ph2_optmean(CMF,1:6),NaN,s_ph2_optmean(CMF,7:end)],[t_ph2_optmean(CMF,1:6),NaN,t_ph2_optmean(CMF,7:end)],'r','LineWidth',2);
% plot([s_ph2_optrms(CMF,1:6),NaN,s_ph2_optrms(CMF,7:end)],[t_ph2_optrms(CMF,1:6),NaN,t_ph2_optrms(CMF,7:end)],'g','LineWidth',2);
% plot([s_optbest(CMF,1:6),NaN,s_optbest(CMF,7:end)],[t_optbest(CMF,1:6),NaN,t_optbest(CMF,7:end)],'c','LineWidth',2);
% plot([s_sl_bl2',NaN,s_pl_bl2'],[t_sl_bl2',NaN,t_pl_bl2'],'k','LineWidth',2);
% title('Spectral & Plankian Loci with CMF# ', CMF);
% legend('Original Transform','Mean-Opt Transform','RMS-Opt Transform','Best-Opt Transform','Baseline (2015 10^{\circ})');
% xlabel('s');
% ylabel('t');
% hold off;
% 
% figure(2)
% hold on;
% for j = 1:size(s_ori2,2)
%     scatter(s_ori2(:,j),t_ori2(:,j),'bx');
% end
% for j = 1:size(spds_ph2,2)-1
%     scatter(s_ph2_optmean(:,j),t_ph2_optmean(:,j),'r.');
%     scatter(s_ph2_optrms(:,j),t_ph2_optrms(:,j),'g.');
%     scatter(s_optbest(:,j),t_optbest(:,j),'y.');
% end
% for j = 1:length(s_bl2)
%     plot(s_bl2(j),t_bl2(j),'k+','MarkerSize',5,'LineWidth',1);
%     text(s_bl2(j)+0.003,t_bl2(j)+0.003,string(opt_data_ph2(1,j+1)));
% end
% h = zeros(5,1);
% h(1) = plot(NaN,NaN,'bx');
% h(2) = plot(NaN,NaN,'r.');
% h(3) = plot(NaN,NaN,'g.');
% h(4) = plot(NaN,NaN,'y.');
% h(5) = plot(NaN,NaN,'k+');
% legend(h,'Original Transform','Mean-Opt Transform','RMS-Opt Transform','Best-Opt Transform','Baseline (2015 10^{\circ})');
% title('Overview');
% xlabel('s');
% ylabel('t');
% hold off;
% 
% fprintf('SPD names:\n');
% disp(opt_data_ph2(1,2:end));
% prompt2 = 'Select an SPD name listed above:  ';
% selectedSPDname = input(prompt2);
% index2 = find(opt_data_ph2(1,2:end) == selectedSPDname);
% figure(3)
% hold on;
% scatter(s_ori2(:,index2),t_ori2(:,index2),'bx');
% scatter(s_ph2_optmean(:,index2),t_ph2_optmean(:,index2),'r.');
% scatter(s_ph2_optrms(:,index2),t_ph2_optrms(:,index2),'g.');
% scatter(s_optbest(:,index2),t_optbest(:,index2),'c.');
% plot(s_bl2(index2),t_bl2(index2),'k+','MarkerSize',5,'LineWidth',1);
% title('Coordinates of SPD name: ', string(selectedSPDname));
% legend('Original Transform','Mean-Opt Transform','RMS-Opt Transform','Best-Opt Transform','Baseline (2015 10^{\circ})');
% xlabel('s');
% ylabel('t');
% hold off;

%% Running Completed
fprintf('Running Completed\n');