% This code is to test if we get a good error if we train on first half,
% test on the second half of each epoch and it seems the case

%load("READY4DL_500.MAT");

%------------- Compare Subjects 1 by 1 : HC vs. PD -------------
sub_no = 1;
for s = 1: sub_no
    [X, Y] = MakeXY(ALL_HC_GVS_OFF, s);
    [Xdmd_predicted, Xdmd_Trained] = myDMD (X,Y);
    
    
    for ch = 2 : 2
       figure(ch), suptitle (strcat("Predicted signal (250-500) for Subject = ", num2str(s), "Channel = ", num2str(ch)));
       ind500 = (ch - 1)* 500 + 1;
       ind250 = (ch - 1)* 250 + 1;
       for tm = 1 : 59
            
           dm   = real(Xdmd_predicted(ind250:ind250+250 - 1,tm));
           or   =      X   (ind500+250:ind500 + 500 - 1,tm);        
           
           subplot(10,6,tm), plot(dm), title (num2str(tm));hold on; plot (or);  legend ("dmd","org");
       end
    end
    
    
    for ch = 2 : 2
       figure(ch+1), suptitle (strcat("Trained signal (0-250) for Subject = ", num2str(s), "Channel = ", num2str(ch)));
       ind500 = (ch - 1)* 500 + 1;
       ind250 = (ch - 1)* 250 + 1;
       for tm = 1 : 59
            
           dm   = real(Xdmd_Trained(ind250:ind250+250 - 1,tm));
           or   =      X   (ind500:ind500 + 250 - 1,tm);        
           
           subplot(10,6,tm), plot(dm), title (num2str(tm));hold on; plot (or); legend ("dmd","org");
       end
    end
    
end
