clear; clc; close all;

load('microstate_sequences.mat');

data = class_combined_gvs_off(1,:);

ms_vals = unique(data);

for ii = 1:length(ms_vals)
   idx_begin = find(data(2:end) == ms_vals(ii) & data(1:end-1) ~= ms_vals(ii))+1;
   idx_end = find(data(2:end) ~= ms_vals(ii) & data(1:end-1) == ms_vals(ii));
   if data(1) == ms_vals(ii)
      idx_begin = [1 idx_begin]; 
   end
   if data(end) == ms_vals(ii)
      idx_end = [idx_end length(data)]; 
   end
   for jj = 1:length(idx_begin)
       MS_data(ii).Epochs{jj} = double(data(idx_begin(jj):idx_end(jj)));
       MS_data(ii).Epoch_len(jj) = length(data(idx_begin(jj):idx_end(jj)));
   end
end

epoch_len_s = cat(2, MS_data.Epoch_len)/500*1000;

fig = figure;
histogram(epoch_len_s);
xlabel('Time in ms');
ylabel('#');
saveas(fig, 'epoch_len_hist.png');


