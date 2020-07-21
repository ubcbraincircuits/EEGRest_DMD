clear; clc; close all;

GoalDatasetName = 'ALL_HC_GVS_OFF';
data_path = '/Users/abhijitc/Documents/Abhijit/NeuroData_Tutor/EEG_patient_id/googledrive-archive/RestData/READY4DL_500.mat';

load(data_path, GoalDatasetName);
rng('default');
plot_path = 'Plots/'; mkdir(plot_path);

fs = 500;
dt = 1/fs;

GoalDataset = eval(GoalDatasetName);
[pt_no, epochs_no] = size(GoalDataset);
r_vals = 50:50:100; nstacks = 50:50:400;
k_fold = 5;

for pt = 1 : 1 %patients
    
    data = cat(3,GoalDataset{pt,:});
    data = zscore(data);
    data = permute(data,[2,1,3]);
    
    reconErrorTrain = NaN(length(nstacks), length(r_vals), k_fold);
    reconErrorTest = NaN(length(nstacks), length(r_vals), k_fold);
    
    k_fold_idx = crossvalind('KFold',epochs_no,k_fold);
    
    for k = 1:k_fold
        
        dataTrain = data(:,:,k_fold_idx ~= k);
        dataTest = data(:,:,k_fold_idx == k);
        
        for r = 1:length(r_vals)
            
            parfor n = 1:length(nstacks)
                try
                    
                    fprintf('Starting CV = %d, nstacks = %d, r = %d ....\n', k, nstacks(n),r_vals(r));
                    
                    %% DMD Computations
                    
                    Xaug = []; Xaug1 = []; Xaug2 = [];
                    for ii = 1:size(dataTrain,3)
                        temp = genTimeShiftEmbedding(dataTrain(:,:,ii), nstacks(n));
                        Xaug1 = cat(2,Xaug1,temp(:,1:end-1));
                        Xaug2 = cat(2,Xaug2,temp(:,2:end));
                        Xaug = cat(3,Xaug,temp);
                    end
                    
                    Xaug_test = [];
                    for ii = 1:size(dataTest,3)
                        Xaug_test = cat(3,Xaug_test,genTimeShiftEmbedding(dataTest, nstacks(n)));
                    end
                    
                    DmdStruct = runLowRankDMD(Xaug1, Xaug2, r_vals(r), dt);
                    
                    tempErr = NaN(1,size(Xaug,3));
                    for ii = 1:size(Xaug,3)
                        [~, tempErr(ii)] = predictDMD(DmdStruct, Xaug(:,:,ii));
                    end
                    reconErrorTrain(n, r, k) = nanmean(tempErr);
                    
                    tempErr = NaN(1,size(Xaug_test,3));
                    for ii = 1:size(Xaug_test,3)
                        [~, tempErr(ii)] = predictDMD(DmdStruct, Xaug_test(:,:,ii));
                    end
                    reconErrorTest(n, r, k) = nanmean(tempErr);
                catch
                    reconErrorTrain(n, r, k) = NaN;
                    reconErrorTest(n, r, k) = NaN;
                end
                
                %% Figures
                
                %         % SVD Plot
                %
                %         fig = figure;
                %         set(fig, 'Position', [0 0 1440 720]);
                %         plotSVD(Xaug, 5);
                %         suptitle(sprintf('SVD Plot: Sub No: %d, Epoch No: %d, nstacks = %d',pt,ep,nstacks(n)));
                %         saveas(fig, fullfile(plot_path,sprintf('SVD_Plot_Sub%d_Epoch%d_nstacks%d.png',pt,ep,nstacks(n))));
                %
                %         % Reconstruction error
                %
                %         fig = figure;
                %         set(fig, 'Position', [0 0 720 720]);
                %         plotCompare(X, real(Xdmd), 1:27, 9, 3)
                %         suptitle(sprintf('Sub No: %d, Epoch No: %d, nstacks = %d, Training Error = %.2f',pt,ep,nstacks(n),reconErrorTrain(n)));
                %         saveas(fig, fullfile(plot_path,sprintf('ReconErrorTrain_Sub%d_Epoch%d_nstacks%d.png',pt,ep,nstacks(n))));
                %
                %         fig = figure;
                %         set(fig, 'Position', [0 0 720 720]);
                %         plotCompare(X_test, real(Xdmd_test), 1:27, 9, 3)
                %         suptitle(sprintf('Sub No: %d, Epoch No: %d, nstacks = %d, Test Error = %.2f',pt,ep,nstacks(n),reconErrorTest(n)));
                %         saveas(fig, fullfile(plot_path,sprintf('ReconErrorTest_Sub%d_Epoch%d_nstacks%d.png',pt,ep,nstacks(n))));
                %
                %         close all;
            end
            %
            %     % Reconstruction error vs nstacks
            %
            %     fig = figure;
            %     subplot(1,2,1);
            %     plot(nstacks, reconErrorTrain, 'ro-','LineWidth',2);
            %     xlabel('nstacks');
            %     ylabel('MSE');
            %     title('Training Error');
            %     subplot(1,2,2);
            %     plot(nstacks, reconErrorTest, 'ro-','LineWidth',2);
            %     xlabel('nstacks');
            %     ylabel('MSE');
            %     title('Test Error');
            %     suptitle(sprintf('Sub No: %d, Epoch No: %d',pt,ep));
            %     saveas(fig, fullfile(plot_path,sprintf('ReconError_vs_nstacks_Sub%d_Epoch%d.png',pt,ep)));
            %
        end
        
    end
    save(sprintf('Error_2020_07_15_Sub%d.mat',pt),'reconErrorTest', 'reconErrorTrain');
end

%% Plot

fig = figure;
subplot(1,2,1);
heatmap(r_vals, nstacks, nanmedian(reconErrorTrain,3));
colormap hot;
xlabel('r - no. of dimensions');
ylabel('nstacks');
title('Training Error');
subplot(1,2,2);
heatmap(r_vals, nstacks, nanmedian(reconErrorTest,3));
colormap hot;
xlabel('r - no. of dimensions');
ylabel('nstacks');
title('Test Error');

%% Functions

function DmdStruct = runLowRankDMD(X,Y, r, dt)

[U, S, V] = svd(X, 'econ');
Ur = U(:, 1:r);
Sr = S(1:r, 1:r);
Vr = V(:, 1:r);

Atilde = Ur'*Y*Vr/Sr;
[W, D] = eig(Atilde);
Phi = Y*Vr/Sr*W;

DmdStruct.r = r;
DmdStruct.dt = dt;
DmdStruct.A = Atilde;
DmdStruct.Phi = Phi;
DmdStruct.Lambda = diag(D);
DmdStruct.Freq = imag(log(diag(D))/dt)/(2*pi);
DmdStruct.omega = log(diag(D))/dt;

end

function Xaug = genTimeShiftEmbedding(X, nstacks)

% construct the augmented, shift-stacked data matrices
Xaug = [];
for st = 1:nstacks
    Xaug = [Xaug; X(:, st:end-nstacks+st)];
end

end

function [Xdmd, reconError] = predictDMD(DmdStruct, X)

Phi = DmdStruct.Phi;
omega = DmdStruct.omega;
r = DmdStruct.r;
dt = DmdStruct.dt;

% Compute DMD mode amplitudes b
x1 = X(:,1);
b = Phi\x1;

% DMD reconstruction
mm1 = size(X, 2); % mm1 = m - 1
time_dynamics = zeros(r, mm1);
t = (0:mm1-1)*dt; % time vector

for iter = 1:mm1
    time_dynamics(:,iter) = (b.*exp(omega*t(iter)));
end

Xdmd = Phi * time_dynamics;

reconError = immse(X, real(Xdmd));

end

function [] = plotSVD(X, n)

[u1,s1,v1]=svd(X,'econ');

subplot(3,1,1), plot(diag(s1)/(sum(diag(s1))),'ro','Linewidth',3); title("Eigen values");
subplot(3,1,2), plot(v1(:,1:n),'Linewidth',2); title("V basis");
subplot(3,1,3), plot(u1(:,1:n),'Linewidth',2); title("U basis");

end


function [] = plotCompare(X, Xdmd, chs, rows, cols)

for ch = chs
    subplot(rows, cols, ch);
    plot(X(ch,:)); hold on;
    plot(Xdmd(ch,:));
    if ch == 1
        legend("X", "Xdmd");
    end
end

end
