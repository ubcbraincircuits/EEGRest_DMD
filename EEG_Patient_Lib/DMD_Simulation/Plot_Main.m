clear; clc; close all;

data(1) = load('Error1_2020_08_26_Test_Epoch_125.00_Sub1.mat');
data(2) = load('Error1_2020_08_26_Test_Epoch_250.00_Sub1.mat');
data(3) = load('Error1_2020_08_26_Test_Epoch_500.00_Sub1.mat');

%% Plot

r_vals_cell = {25:25:50,50:50:100,100:100:200};
nstacks_cell = {25:25:75, 50:50:150, 100:100:300};

RE = [125, 250, 500];
Fs = 500;

Str = {'Entire Epoch', 'Non time-delay embedding epoch'};

for re = 1:length(RE)
    
    r_vals = r_vals_cell{re};
    nstacks = nstacks_cell{re};
    
    fig = figure;
    set(fig, 'Position', [0 0 500 900]);
    
    for ii = 1:2
        
        subplot(2,2,(ii-1)*2+1);
        h(ii,1) = heatmap(r_vals, nstacks, nanmedian(data(re).reconErrorTrain(:,:,:,re,ii+1),3));
        colormap hot;
        xlabel('r - no. of dimensions');
        ylabel('nstacks');
        title(sprintf('Training Error \n %s',Str{ii}));
        subplot(2,2,(ii-1)*2+2);
        h(ii,2) = heatmap(r_vals, nstacks, nanmedian(data(re).reconErrorTest(:,:,:,re,ii+1),3));
        colormap hot;
        xlabel('r - no. of dimensions');
        ylabel('nstacks');
        title(sprintf('Test Error \n %s',Str{ii}));
        
    end
    [h.ColorLimits] = deal([0.55 0.95]);
    suptitle(sprintf('Length of epoch \n %0.2fs',RE(re)/Fs));
    saveas(fig, sprintf('Error_plot_epoch%d.png',RE(re)));
end
