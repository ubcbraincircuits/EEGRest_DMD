clear; clc; close all;
load('/Users/abhijitc/Documents/Abhijit/NeuroData_Tutor/EEG_patient_id/googledrive-archive/RestData/READY4DL_500.mat','ALL_HC_GVS_OFF','ALL_PD_GVSOFF_MEDOFF');
rng('default');
plot_path = 'Plots/'; mkdir(plot_path);

sub_no = 1;
for s = sub_no
    data = cat(3,ALL_HC_GVS_OFF{s,:});
    data = zscore(data);
    data = permute(data, [2,1,3]);
    
    idx = [ones(1,250), 2*ones(1,250)];%crossvalind('KFold',size(data,3),2);
    
    X_train = data(:,idx == 1,1:end-1); Y_train = data(:,idx == 1,2:end);
    X_test = data(:,idx == 2,1:end-1); Y_test = data(:,idx == 2,2:end);
    
    [n_ch, n_t, n_ep] = size(X_train);
    
    DmdStruct = runDMD(reshape(X_train, [n_ch*n_t, n_ep]),reshape(Y_train, [n_ch*n_t, n_ep]));
    
    Y_train_predict = predictDMD(X_train(:,:,1), DmdStruct, n_ep);
    Y_test_predict = predictDMD(X_test(:,:,1), DmdStruct, n_ep);
    
    ch = 2;
    
    fig = figure;
    set(fig, 'Position', [0 0 1440 720]);
    plotChComp(Y_train, Y_train_predict, ch, [10,6]);
    suptitle(sprintf('Train Epochs : Sub No = %d, Ch = %d',s,ch));
    saveas(fig,fullfile(plot_path, sprintf('Train_Epochs_SubNo%d_Ch%d.png',s,ch)));
    
    fig = figure;
    set(fig, 'Position', [0 0 1440 720]);
    plotChComp(Y_test, Y_test_predict, ch, [10,6]);
    suptitle(sprintf('Test Epochs : Sub No = %d, Ch = %d',s,ch));
    saveas(fig,fullfile(plot_path, sprintf('Test_Epochs_SubNo%d_Ch%d.png',s,ch)));
    
end

%% Functions

function DmdStruct = runDMD(X1,Y1)

% SVD and truncate to first r modes
[U, S, V] = svd(X1, 'econ');

%figure (1), semilogy(diag(S))
r = 59;
DmdStruct.r = r;

Ur = U(:, 1:r);
Sr = S(1:r, 1:r);
%figure (2), plot(diag(Sr)/sum(diag(Sr)), 'ro');

Vr = V(:, 1:r);
% figure (3);
% subplot(3,1,1), plot(Ur(1:1000, 1:5));
% subplot(3,1,2), plot(Vr(1:59, 1:5));

% DMD modes
DmdStruct.Atilde = Ur'*Y1*Vr/Sr;
[DmdStruct.W, DmdStruct.D] = eig(DmdStruct.Atilde);
DmdStruct.Phi = Y1*Vr/Sr*DmdStruct.W;
% figure (3);
% subplot(3,1,3), plot(real(Phi(1:1000, 1:3)));
%
% DMD eigenvalues
DmdStruct.dt = 1; %20 milisec
DmdStruct.lambda = diag(DmdStruct.D);
DmdStruct.omega = log(DmdStruct.lambda)/DmdStruct.dt/2/pi;

end

function Y = predictDMD(X, DmdStruct, m)

[xd,yd,~] = size(X);
x1 = reshape(X,[xd*yd, 1]);

b = DmdStruct.Phi\x1;
time_dynamics = zeros(DmdStruct.r, m);
t = (1:m)/DmdStruct.dt;
for iter = 1:m
    time_dynamics(:,iter) = (b.*exp(DmdStruct.omega*t(iter)));
end

Y_temp = DmdStruct.Phi * time_dynamics;
Y = reshape(abs(Y_temp),[xd,yd,m]);

end

function [] = plotChComp(Y_orig, Y_predict, ch, RowCol)
    for ii = 1:size(Y_orig, 3)
       subplot(RowCol(1),RowCol(2),ii);
       plot(Y_orig(ch,:,ii)); hold on;
       plot(Y_predict(ch,:,ii)); 
       title(sprintf('Epoch = %d, r = %.2f',ii, corr(Y_orig(ch,:,ii)',Y_predict(ch,:,ii)')));
       legend({'Orig','DMD'});
    end
end