clear; clc; close all;

load('Error_2020_07_21_Epoch_0.5_Sub1.mat');
r_vals = 50:50:100; nstacks = 50:50:150;

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