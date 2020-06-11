function [X,Y] = MakeXY(data, subject)
all_trials_X = [];
% all_trials_Y = [];

[~ , epochs] = size(data);
for t = 1: epochs
    a = data{subject,t};
    b = reshape(a(1:500,:), [500 * 27,1]);
    all_trials_X = [all_trials_X  b];
end

[X_whitened, mu, invMat, whMat] = whiten(all_trials_X);
    
X = X_whitened(:,1:epochs-1);
Y = X_whitened(:,2:epochs);

end