% This code is to test if we get a good error if we train on first half,
% test on the second half of each epoch and it seems the case

%load("READY4DL_500.MAT");

%------------- Compare Subjects 1 by 1 : HC vs. PD -------------
sub_no = 1;
for s = 1: sub_no
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
    
    for ch = 2 : 2
        fig = figure;
        set(fig, 'Position', [0 0 1440 720]);
        suptitle (strcat("Predicted signal (250-500) for Subject = ", num2str(s), "Channel = ", num2str(ch)));
        ind500 = (ch - 1)* 500 + 1;
        ind250 = (ch - 1)* 250 + 1;
        for tm = 1 : 59
            
            dm   = real(Xdmd_predicted(ind250:ind250+250 - 1,tm));
            or   = Y(ind500+250:ind500 + 500 - 1,tm);
            
            subplot(10,6,tm);
            plot(dm);
            hold on; plot(or);
            legend("dmd","org");
            title(sprintf('%d: r = %.2f',tm,corr(dm,or)));
        end
    end
    
    
    for ch = 2 : 2
        fig = figure;
        set(fig, 'Position', [0 0 1440 720]);
        suptitle (strcat("Trained signal (0-250) for Subject = ", num2str(s), "Channel = ", num2str(ch)));
        ind500 = (ch - 1)* 500 + 1;
        ind250 = (ch - 1)* 250 + 1;
        for tm = 1 : 59
            
            dm   = real(Xdmd_Trained(ind250:ind250+250 - 1,tm));
            or   = Y(ind500:ind500 + 250 - 1,tm);
            
            subplot(10,6,tm);
            plot(dm);
            hold on; plot(or);
            legend("dmd","org");
            title(sprintf('%d: r = %.2f',tm,corr(dm,or)));
        end
    end
    
end
