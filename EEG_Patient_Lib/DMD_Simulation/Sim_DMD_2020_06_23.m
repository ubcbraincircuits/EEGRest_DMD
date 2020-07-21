clc; clear; close all;

rng('default');

%% Set latent modes

n_ch = 27; n_latent_dim = 2; 
Freq = [5, 13];
Phase = rand(1,n_latent_dim)*2*pi;
Amp = 1./Freq;
Phi = rand(n_ch, n_latent_dim)-0.5;
T = 1; Fs = 250;
t = (1:T*Fs)/Fs;

sig = 0;
X_orig = Phi*(Amp'.*cos(2*pi*Freq'*t + Phase')) + sig*rand(n_ch, length(t));

X_complex = hilbert(X_orig);

plot(t, real(X_complex'));

%% Perform DMD

lag = 1;
X = X_complex(:,1:end-lag);
Y = X_complex(:,lag+1:end);

DmdExactStruct = runExactDMD(X,Y,lag/Fs);
DmdLowRankStruct = runLowRankDMD(X,Y, 2, lag/Fs);

%% Functions

function DmdStruct = runExactDMD(X,Y, del_t)
    A = Y*pinv(X);
    [W, Lambda] = eig(A);
    
    DmdStruct.A = A;
    DmdStruct.Phi = W;
    DmdStruct.Lambda = diag(Lambda);
    DmdStruct.Freq = imag(log(diag(Lambda))/del_t)/(2*pi);
end

function DmdStruct = runLowRankDMD(X,Y, r, del_t)

[U, S, V] = svd(X, 'econ');
Ur = U(:, 1:r);
Sr = S(1:r, 1:r);
Vr = V(:, 1:r);

Atilde = Ur'*Y*Vr/Sr;
[W, D] = eig(Atilde);
Phi = Y*Vr/Sr*W;

DmdStruct.A = Atilde;
DmdStruct.Phi = Phi;
DmdStruct.Lambda = diag(D);
DmdStruct.Freq = imag(log(diag(D))/del_t)/(2*pi);
DmdStruct.Freq = log(diag(D))/del_t;

end

