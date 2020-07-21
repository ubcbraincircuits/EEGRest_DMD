function [lambda, omega, f, P] = myDMD2 (X,Y)

ss = size(X,1);
X1 = X(1:ss/2,:);
X2 = X(ss/2+1:end, :);

Y1 = Y(1:ss/2,:);
Y2 = Y(ss/2+1:end, :);

%% let's full data be used
X1 = X;
Y1 = Y;

%%
% SVD and truncate to first r modes
[U, S, V] = svd(X1, 'econ');

%figure (1), semilogy(diag(S))

r = 59;

U_r = U(:, 1:r);
S_r = S(1:r, 1:r);
%figure (2), plot(diag(Sr)/sum(diag(Sr)), 'ro');

V_r = V(:, 1:r);
% figure (3);
% subplot(3,1,1), plot(Ur(1:1000, 1:5));
% subplot(3,1,2), plot(Vr(1:59, 1:5));

% DMD modes 
Atilde = U_r'*Y1*V_r/S_r;
[W, D] = eig(Atilde);
Phi = Y1*V_r/S_r*W;
% figure (3);
% subplot(3,1,3), plot(real(Phi(1:1000, 1:3)));
% 
% DMD eigenvalues
dt = 1; %20 milisec
lambda = diag(D);
omega = log(lambda)/dt/2/pi; 
%%
Ahat = (S_r^(-1/2)) * Atilde * (S_r^(1/2));
[What, D] = eig(Ahat);
W_r = S_r^(1/2) * What;
Phi = Y1*V_r/S_r*W_r;

f = abs(imag(omega));
P = (diag(Phi'*Phi));

% %% DMD reconstructions (PREDICTED)
% x1 = X2(:, 1);
% b = Phi\x1;
% 
% m = size(X1, 2);
% time_dynamics = zeros(r, m);
% t = (1:m)/dt;
% for iter = 1:m,
%     time_dynamics(:,iter) = (b.*exp(omega*t(iter)));
% end;
% Xdmd_predicted = Phi * time_dynamics;
% 
% %% DMD reconstructions (TRAINED)
% x1 = X1(:, 1);
% b = Phi\x1;
% 
% m = size(X1, 2);
% time_dynamics = zeros(r, m);
% t = (1:m)/dt;
% for iter = 1:m,
%     time_dynamics(:,iter) = (b.*exp(omega*t(iter)));
% end;
% Xdmd_trained = Phi * time_dynamics;

end