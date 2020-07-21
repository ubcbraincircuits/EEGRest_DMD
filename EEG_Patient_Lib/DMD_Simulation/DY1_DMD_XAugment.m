
%load READY4DL_500.MAT;

all = [];
fs = 500;

%GoalDatasetName =  "ALL_PD_GVSOFF_MEDOFF";
%GoalDatasetName = "ALL_PD_GVSOFF_MEDON";
%GoalDatasetName = "ALL_PD_GVSON_MEDOFF";
%GoalDatasetName = "ALL_PD_GVSON_MEDON";

%GoalDatasetName = "ALL_HC_GVS_ON";
GoalDatasetName = "ALL_HC_GVS_OFF";

GoalDataset = eval(GoalDatasetName);

[pt_no epochs_no] = size(GoalDataset);
for pt = 1 : 1 %patients
   pt
   for ep = 1 : 1 %epochs 
      one = GoalDataset{pt,ep}';

      dt = 1/fs;
      X = one;
      X1 = X;
      
      [u1,s1,v1]=svd(X,'econ');
      figure,
      suptitle(strcat("SVD on X, patient = ", num2str(pt), ", trial =", num2str(ep)));
      subplot(2,1,1), plot(diag(s1)/(sum(diag(s1))),'ro','Linewidth',3), title("Eigen values")
      subplot(2,1,2), plot(v1(1:100,1:5),'Linewidth',2),                 title("V basis")

      r = 50; 
      nstacks = 38; 

      % construct the augmented, shift-stacked data matrices
      Xaug = [];
      for st = 1:nstacks
         Xaug = [Xaug; X1(:, st:end-nstacks+st)];
      end
      

      [u1,s1,v1]=svd(Xaug,'econ');
      figure,
      suptitle(strcat("SVD on XAug, patient = ", num2str(pt), ", trial =", num2str(ep)));
      subplot(2,1,1), plot(diag(s1)/(sum(diag(s1))),'ro','Linewidth',3),title("Eigen values")
      subplot(2,1,2), plot(v1(1:100,1:5),'Linewidth',2),   title("V basis")            

      % SVD and truncate to first r modes
      
      X = Xaug(:, 1:end-1);
      Y = Xaug(:, 2:end);
      
      [U, S, V] = svd(X, 'econ');
      U_r = U(:, 1:r);
      S_r = S(1:r, 1:r);
      V_r = V(:, 1:r);

     % DMD modes 
      Atilde = U_r'*Y*V_r/S_r;
      [W_r, D] = eig(Atilde);
      Phi = Y*V_r/S_r*W_r;

     %DMD eigenvalues
     lambda = diag(D);
     omega = log(lambda)/dt/2/pi;
    
    %% eigenvalue
    figure,
    subplot(1,2,1);
    plot(lambda, 'k.');
    rectangle('Position', [-1 -1 2 2], 'Curvature', 1, ...
        'EdgeColor', 'k', 'LineStyle', '--');
    axis(1.2*[-1 1 -1 1]);
    axis square;


    subplot(1,2,2);
    plot(omega, 'k.');
    line([0 0], 200*[-1 1], 'Color', 'k', 'LineStyle', '--');
    axis([-8 2 -170 +170]);
    axis square;


    %% spectrum
    % alternate scaling of DMD modes
    Ahat = (S_r^(-1/2)) * Atilde * (S_r^(1/2));
    [What, D] = eig(Ahat);
    W_r = S_r^(1/2) * What;
    Phi = Y*V_r/S_r*W_r;

    f = abs(imag(omega));
    P = (diag(Phi'*Phi));

    % DMD spectrum
    figure,
    subplot(1,2,1);
    stem(f, P, 'k');
    xlim([0 150]);
    axis square;

    % power spectrum
    timesteps = size(X, 2);
    srate = 1/dt;
    nelectrodes = 27;
    NFFT = 2^nextpow2(timesteps);
    f = srate/2*linspace(0, 1, NFFT/2+1);

    subplot(1,2,2); 
    hold on;
    for c = 1:nelectrodes,
        fftp(c,:) = fft(X(c,:), NFFT);
        plot(f, 2*abs(fftp(c,1:NFFT/2+1)), ...
            'Color', 0.6*[1 1 1]);
    end;
    plot(f, 2*abs(mean(fftp(c,1:NFFT/2+1), 1)), ...
        'k', 'LineWidth', 2);
    xlim([0 100]);
    ylim([0 60]);
    axis square;
    box on;
    
    
    
    %% Compute DMD mode amplitudes b
    x1 = X(:, 1);
    b = Phi\x1;

    %% DMD reconstruction
    mm1 = size(X, 2); % mm1 = m - 1
    time_dynamics = zeros(r, mm1);
    t = (0:mm1-1)*dt; % time vector
    for iter = 1:mm1,
        time_dynamics(:,iter) = (b.*exp(omega*t(iter)));
    end;
    Xdmd = Phi * time_dynamics;
    
   figure,
   suptitle(strcat("X vs. Xdmd, patient = ", num2str(pt), ", trial =", num2str(ep)));
   plot(Y(1,:)),hold on, plot(real(Xdmd(1,:))), legend("X", "Xdmd");

   end
            

   
end

