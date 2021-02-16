clear; clc; close all;

n_s = 22;
for s = 1:n_s
    a = load(sprintf('Results_re1_Sub%d.mat',s));
    dataTest(:,s,:) = cat(1,a.Out_Struct.reconErrorTest);
    dataTrain(:,s,:) = cat(1,a.Out_Struct.reconErrorTrain);
    disp(s);
end

%% Plots

close all;
fig = figure;
set(fig, 'Position',[0 0 1440 900]);
subplot(2,1,1);
bar(1:22,nanmean(dataTrain(:,:,3),1));
hold on;
errorbar(1:22,nanmean(dataTrain(:,:,3),1),nanstd(dataTrain(:,:,3),[],1)./sqrt(n_s),'.');
ylim([0.4 1.2]);
title('Train data');
xlabel('Subject #'); ylabel('Reconstruction error');
subplot(2,1,2);
bar(1:22,nanmean(dataTest(:,:,3),1));
hold on;
errorbar(1:22,nanmean(dataTest(:,:,3),1),nanstd(dataTest(:,:,3),[],1)./sqrt(n_s),'.');
ylim([0.4 1.2]);
title('Test data');
xlabel('Subject #'); ylabel('Reconstruction error');
set(findall(gcf,'-property','FontSize'),'FontSize',20);
saveas(fig,sprintf('Error_DMD.png'));

%% Compare

for ii = 1:4
   fig = figure;
   set(fig,'Position',[0 0 1440 900]);
   X = revTimeShiftEmbedding(a.Out_Struct(1).X_test,150);
   Xdmd = revTimeShiftEmbedding(abs(a.Out_Struct(1).Xdmd_test),150);
   for jj = 1:27
       subplot(10,3,jj);
       h(1) = plot((1:size(X,2))/500,X(jj,:),'b-','LineWidth',1);
       hold on; 
       h(2) = plot((1:size(X,2))/500,Xdmd(jj,:),'r-','LineWidth',1);
       title(sprintf('Ch %d',jj));
   end
   legend(h,{'Original','DMD'},'Location','BestOutside');
   %set(findall(gcf,'-property','FontSize'),'FontSize',20);
   saveas(fig,sprintf('X_Xdmd_plot%d.png',ii));
end

%% Function

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
