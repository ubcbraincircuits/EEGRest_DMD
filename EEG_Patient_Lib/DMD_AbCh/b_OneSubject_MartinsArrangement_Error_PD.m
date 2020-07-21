% This is to check how error changes for subjects and UPDRS
%load("READY4DL_500.MAT");
%------------- Compare Subjects 1 by 1 : HC vs. PD -------------
errors_HC = zeros(22, 27, 60);

for s = 1: 22
    [X, Y] = MakeXY(ALL_HC_GVS_OFF, s);
    [Xdmd_predicted, Xdmd_Trained] = myDMD (X,Y);
    
    Xdmd_predicted = reshape(Xdmd_predicted, [27,250,59]);
    Xdmd_predicted = permute(Xdmd_predicted,[2,1,3]);
    Xdmd_predicted = reshape(Xdmd_predicted, [27*250,59]);
    
    Xdmd_Trained = reshape(Xdmd_Trained, [27,250,59]);
    Xdmd_Trained = permute(Xdmd_Trained,[2,1,3]);
    Xdmd_Trained = reshape(Xdmd_Trained, [27*250,59]);
    
    Y = reshape(Y, [27,500,59]);
    Y = permute(Y,[2,1,3]);
    Y = reshape(Y, [27*500,59]);
    
    for ch = 1 : 27
       ind500 = (ch - 1)* 500 + 1;
       ind250 = (ch - 1)* 250 + 1;
       for tm = 1 : 59
            
           dm   = real(Xdmd_predicted(ind250    :ind250 + 250 - 1,tm));
           or   = Y(ind500+250:ind500 + 500 - 1,tm);        
           errors_HC(s,ch,tm) = immse(dm, or);
          
       end
    end
    
end

%% 
errors_PD = zeros(20, 27, 60);

for s = 1: 20
    [X, Y] = MakeXY(ALL_PD_GVSOFF_MEDOFF, s);
    [Xdmd_predicted, Xdmd_Trained] = myDMD (X,Y);
    
    for ch = 1 : 27
       ind500 = (ch - 1)* 500 + 1;
       ind250 = (ch - 1)* 250 + 1;
       for tm = 1 : 59
            
           dm   = real(Xdmd_predicted(ind250    :ind250 + 250 - 1,tm));
           or   =      X             (ind500+250:ind500 + 500 - 1,tm);        
           errors_PD(s,ch,tm) = immse(dm, or);
          
       end
    end
    
end

%% 
UPDRS = [26 17 11 37 29 8 39 35 12 30 13 31 34 32 14 21 29 20 18 14];
er_pd = mean(mean(errors_PD, 3), 2); yyaxis left, plot(er_pd); yyaxis right, plot(UPDRS), legend ("Average DMD Recon. Error over channels-Trials",  "UPDRS")
xlabel ("PD Subjects")
[Corr_val, p_value] = corrcoef(er_pd, UPDRS)

inds = find(UPDRS>=31);
H_UPDRS = UPDRS(inds);
H_err   = er_pd(inds);


inds = find(UPDRS < 15);
L_UPDRS = UPDRS(inds);
L_err  = er_pd(inds);


corrcoef(H_err, H_UPDRS)
corrcoef(L_err, L_UPDRS)
corr2(er_pd, UPDRS')
