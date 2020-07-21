clear; clc; close all;
load('/Users/abhijitc/Documents/Abhijit/NeuroData_Tutor/EEG_patient_id/googledrive-archive/RestData/READY4DL_500.mat','ALL_HC_GVS_OFF');
rng('default');
plot_path = 'Plots/'; mkdir(plot_path);

sub_no = 1;
for s = sub_no
    data = cat(3,ALL_HC_GVS_OFF{s,:});
    data = zscore(data);
    data = permute(data, [2,1,3]);
    
    idx = [ones(1,30), 2*ones(1,30)];%crossvalind('KFold',size(data,3),2);
    
    X_train = data(:,1:end-1,idx==1); Y_train = data(:,2:end,idx==1);
    X_test = data(:,1:end-1,idx==2); Y_test = data(:,2:end,idx==2);
    
    [n_ch, n_t, n_ep] = size(X_train);
    
    A = runDMD(reshape(X_train, [n_ch, n_t*n_ep]),reshape(Y_train, [n_ch, n_t*n_ep]));
    
    Y_train_predict = predictDMD(X_train(:,1,:), A, n_t);
    Y_test_predict = predictDMD(X_test(:,1,:), A, n_t);
    
%     Y_train_predict = predictDMD2(X_train, A);
%     Y_test_predict = predictDMD2(X_test, A);
    
    ch = 2;
    
    fig = figure;
    set(fig, 'Position', [0 0 1440 720]);
    plotChComp(Y_train, Y_train_predict, ch, [6,5], find(idx == 1));
    suptitle(sprintf('Train Epochs : Sub No = %d, Ch = %d',s,ch));
    saveas(fig,fullfile(plot_path, sprintf('Train_Epochs_SubNo%d_Ch%d.png',s,ch)));
    
    fig = figure;
    set(fig, 'Position', [0 0 1440 720]);
    plotChComp(Y_test, Y_test_predict, ch, [6,5], find(idx == 2));
    suptitle(sprintf('Test Epochs : Sub No = %d, Ch = %d',s,ch));
    saveas(fig,fullfile(plot_path, sprintf('Test_Epochs_SubNo%d_Ch%d.png',s,ch)));
    
end

%% Functions

function [A] = runDMD(X,Y)
    A = Y*pinv(X);
end

function Y = predictDMD(X, A, n_t)

[xd,~,zd] = size(X);
X_temp = reshape(X,[xd, zd]);

for t = 1:n_t
    Y_temp(:,:,t) = (A^(t))*X_temp;
end

Y = permute(Y_temp,[1,3,2]);

end

function Y = predictDMD2(X, A)

[xd,yd,zd] = size(X);
X_temp = reshape(X,[xd, yd*zd]);
Y_temp = A*X_temp;
Y = reshape(Y_temp,[xd,yd,zd]);

end

function [] = plotChComp(Y_orig, Y_predict, ch, RowCol, EpNum)
    for ii = 1:size(Y_orig, 3)
       subplot(RowCol(1),RowCol(2),ii);
       plot(Y_orig(ch,:,ii)); hold on;
       plot(Y_predict(ch,:,ii)); 
       title(sprintf('Epoch = %d, r = %.2f',EpNum(ii),corr(Y_orig(ch,:,ii)',Y_predict(ch,:,ii)')));
       legend({'Orig','DMD'});
    end
end