function [CCTst,Dst] = CCTst_Dst_2015_10(p,q,s_test,t_test,,s_ct_bl_opt,t_ct_bl_opt)

CT_temps_baseline = xlsread('CT_temps.xlsx');
CT_baseline = CT_temps_baseline(:,1);
Xb = xlsread('1000CMFs.xlsx', 'Xb_2015');
Yb = xlsread('1000CMFs.xlsx', 'Yb_2015');
Zb = xlsread('1000CMFs.xlsx', 'Zb_2015');


for p = 1:size(Xb,2)
    for q = 1:2
        d_in(q) = sqrt((s_test - s_ct_bl_opt(p,q))^2 + (t_test - t_ct_bl_opt(p,q))^2);
    end

    for q = 3:length(CT)+1
        d_in(q) = sqrt((s_test - s_ct_bl_opt(p,q))^2 + (t_test - t_ct_bl_opt(p,q))^2);
        if d_in(q-2) > d_in(q-1) & d_in(q-1) < d_in(q)
            break
        end
    end

    q = q-1;

    for m = 1:3
        cctst(m) = CT_temps_baselilne(q+m-2);
        s(m) = s_ct_bl_opt(p,q+m-2);
        t(m) = t_ct_bl_opt(p,q+m-2);
        d(m) = sqrt((s_test - s_ct_bl_opt(p,q+m-2))^2 + (t_test - t_ct_bl_opt(p,q+m-2))^2);
    end

    %Triangular Solution
    d_tri = sqrt((s(3)-s(1))^2 + (t(3)-t(1))^2);
    x = (d(1)^2 - d(3)^2 + d_tri^2) / (2 * d_tri);
    CCTst_triangular = cctst(1) + (cctst(3)-cctst(1))*x / d_tri;
    Dst_triangular_dist = sqrt(d(1)^2 - x^2);

    %Dst Sign
    t_Planck = t(1) + (t(3)-t(1))*x / d_tri;
    if t_test > t_Planck
       sign = 1;
    else
        sign = -1;
    end

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

    CCTst = Calc_CCTst;
    Dst = Calc_Dst;
end

end