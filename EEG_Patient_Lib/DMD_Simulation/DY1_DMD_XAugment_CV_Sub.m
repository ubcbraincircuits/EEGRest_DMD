clear; clc; close all;

Type = {'ALL_HC_GVS_OFF','ALL_HC_GVS_ON','ALL_PD_GVSOFF_MEDOFF',...
    'ALL_PD_GVSOFF_MEDON','ALL_PD_GVSON_MEDOFF','ALL_PD_GVSON_MEDON'};

for main_loop = 1:length(Type)
    
    clearvars -except Type main_loop;
    
    GoalDatasetName = Type{main_loop};
    data_path = '/Users/abhijitc/Documents/Abhijit/NeuroData_Tutor/EEG_patient_id/READY4DL_500.mat';
    
    load(data_path, GoalDatasetName);
    rng('default');
    save_path = sprintf('Results_2_%d/',GoalDatasetName); mkdir(save_path);
    
    fs = 500;
    dt = 1/fs;
    
    GoalDataset = eval(GoalDatasetName);
    [pt_no, epochs_no] = size(GoalDataset);
    % r_vals = 100; nstacks = 150;
    r_vals_cell = {50};
    nstacks_cell = {150};
    
    nPl = 4;
    reEpoch = [250];
    
    for pt = 1 : pt_no
        
        fprintf('Starting Sub = %d ....\n', pt);
        
        for re = 1:length(reEpoch)
            
            r_vals = r_vals_cell{re};
            nstacks = nstacks_cell{re};
            
            data = cat(1,GoalDataset{pt,:}); % data = 30000 x 27
            nCh = size(GoalDataset{pt,1}, 2);
            nSamples = size(GoalDataset{pt,1},1);
            nTrials = length(GoalDataset(pt,:));
            
            data = whiten(data);
            data = reshape(data, [nSamples, nTrials, nCh]); % data = 500 x 60 x 27
            
            data = permute(data,[3,1,2]);
            data = reshape(data, [nCh, reEpoch(re), nTrials*nSamples/reEpoch(re)]);
            
            nepoch = size(data,3);
            reconErrorTrain = NaN(length(nstacks), length(r_vals), nepoch, length(reEpoch), 3);
            reconErrorTest = NaN(length(nstacks), length(r_vals), nepoch, length(reEpoch), 3);
            
            parfor k = 1:nepoch
                %     for k = [30, 60, 90, 120]
                Out_Struct(k) = loopfun(data, k, r_vals, nstacks, nepoch, dt);
            end
            
            error = squeeze(cat(4,Out_Struct.reconErrorTest));
            [~, max_idx] = max(error(1,:));
            MaxStruct = Out_Struct(max_idx);
            
            [~, min_idx] = min(error(1,:));
            MinStruct = Out_Struct(min_idx);
            
            save([save_path,sprintf('Results_re%d_Sub%d.mat',re,pt)],'MaxStruct','MinStruct');
            clear Out_Struct;
        end
    end
    
end

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

function X = revTimeShiftEmbedding(Xaug, nstacks)

ch = size(Xaug,1)/nstacks;
cols = size(Xaug, 2);
T = cols + nstacks - 1;

X_big = NaN(ch,T,cols);

for col = 1:cols
    X_big(:,col:col+nstacks-1,col) = reshape(Xaug(:,col),ch,nstacks);
end

X = nanmean(X_big, 3);

end

function [Xdmd, reconError] = predictDMD(DmdStruct, X, nstacks)

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

reconError(1) = immse(X, real(Xdmd))./immse(X, zeros(size(X)));

X = revTimeShiftEmbedding(X, nstacks);
Xdmd1 = revTimeShiftEmbedding(real(Xdmd), nstacks);

reconError(2) = immse(X, Xdmd1)./immse(X, zeros(size(X)));
reconError(3) = immse(X(:,nstacks+1:end), Xdmd1(:,nstacks+1:end))./...
    immse(X(:,nstacks+1:end), zeros(size(X(:,nstacks+1:end))));

end

function Out_Struct = loopfun(data, k, r_vals, nstacks, nepoch, dt)

dataTrain = data(:,:,setdiff(1:nepoch, k));
dataTrainMat{1} = reshape(data(:,:,1:k-1),size(data,1),[]);
dataTrainMat{2} = reshape(data(:,:,k+1:end),size(data,1),[]);

dataTest = data(:,:,k);

for r = 1:length(r_vals)
    
    for n = 1:length(nstacks)
        
        %% DMD Computations
        
        Xaug = []; Xaug1 = []; Xaug2 = [];
        for ii = 1:2
            temp = genTimeShiftEmbedding(dataTrainMat{ii}, nstacks(n));
            Xaug1 = cat(2,Xaug1,temp(:,1:end-1));
            Xaug2 = cat(2,Xaug2,temp(:,2:end));
        end
        
        for ii = 1:size(dataTrain,3)
            temp = genTimeShiftEmbedding(dataTrain(:,:,ii), nstacks(n));
            Xaug = cat(3,Xaug,temp);
        end
        
        Xaug_test = [];
        for ii = 1:size(dataTest,3)
            Xaug_test = cat(3,Xaug_test,genTimeShiftEmbedding(dataTest, nstacks(n)));
        end
        
        Out_Struct.DmdStruct(n,r) = runLowRankDMD(Xaug1, Xaug2, r_vals(r), dt);
        
        tempErr = NaN(3,size(Xaug,3));
        for ii = 1:size(Xaug,3)
            [~, tempErr(:,ii)] = predictDMD(Out_Struct.DmdStruct(n,r), Xaug(:,:,ii), nstacks(n));
        end
        Out_Struct.reconErrorTrain(n, r, :) = nanmean(tempErr,2);
        
        tempErr = NaN(3,size(Xaug_test,3));
        for ii = 1:size(Xaug_test,3)
            [Xdmd(:,:,ii), tempErr(:,ii)] = predictDMD(Out_Struct.DmdStruct(n,r), Xaug_test(:,:,ii), nstacks(n));
        end
        Out_Struct.reconErrorTest(n, r, :) = nanmean(tempErr,2);
        
        Out_Struct.X_test = Xaug_test;
        Out_Struct.Xdmd_test = Xdmd;
        
    end
end
end
