
%load("READY4DL_500.MAT");

%% 

R_Omegas_PD = [];
PH_Omegas_PD = [];
Ps_PD = [];
R_Lambdas_PD = [];
PH_Lambdas_PD = [];


for s = 1: 20
    [X, Y] = MakeXY(ALL_PD_GVSOFF_MEDOFF, s);
    [lambda, omega, f, P] = myDMD2 (X,Y);
    
    R_Lambdas_PD  = [R_Lambdas_PD abs(lambda)];
    PH_Lambdas_PD = [PH_Lambdas_PD angle(lambda)];
    R_Omegas_PD = [R_Omegas_PD abs(omega)];
    PH_Omegas_PD = [PH_Omegas_PD angle(omega)];
    Ps_PD = [Ps_PD P];
    
end

%% 
PH_Lambdas_PD =  PH_Lambdas_PD';
R_Lambdas_PD  =  R_Lambdas_PD';
R_Omegas_PD   =  R_Omegas_PD';
PH_Omegas_PD  =  PH_Omegas_PD';
Ps_PD         =  Ps_PD';



UPDRS = [26 17 11 37 29 8 39 35 12 30 13 31 34 32 14 21 29 20 18 14];

H_inds = find(UPDRS>=31);
H_UPDRS = UPDRS(H_inds);

L_inds = find(UPDRS < 15);
L_UPDRS = UPDRS(L_inds);

%------------------------------------
LPH_OM = zeros(59,1);
HPH_OM = zeros(59,1);

LPH_LAM = zeros(59,1);
HPH_LAM = zeros(59,1);

LP = zeros(59,1);
HP = zeros(59,1);


for i = 1: 59

%-------------------------------------------------
temp = PH_Omegas_PD(:,i);

templ = corrcoef(temp(L_inds), L_UPDRS);
LPH_OM(i,1) = templ(2);

temph = corrcoef(temp(H_inds), H_UPDRS);
HPH_OM(i,1) = temph(2);

%  yyaxis left, plot(temp(L_inds))
%  yyaxis right, plot(L_UPDRS);

%---------------------------------------------------
temp = PH_Lambdas_PD(:,i);

templ = corrcoef(temp(L_inds), L_UPDRS);
LPH_LAM(i,1) = templ(2);

temph = corrcoef(temp(H_inds), H_UPDRS);
HPH_LAM(i,1) = temph(2);
%--------------------------------------------------
temp = Ps_PD(:,i);

templ = corrcoef(temp(L_inds), L_UPDRS);
LP(i,1) = templ(2);

temph = corrcoef(temp(H_inds), H_UPDRS);
HP(i,1) = temph(2);

end

figure (1), plot(LP); hold on; plot(HP); legend("Low P", "High P"); suptitle ("Correlation with UPDRS: >= 31 or <15");
figure (2), plot(LPH_LAM);hold on; plot(HPH_LAM); legend("Low PH-LAMBDA", "High PH-LAMBDA");suptitle ("Correlation with UPDRS: >= 31 or <15");
figure (3), plot(LPH_OM);hold on; plot(HPH_OM); legend("Low PH-OMEGA", "High PH-OMEGA");suptitle ("Correlation with UPDRS: >= 31 or <15");


%------------------------------------- OMEGA-------------------------------
[rho,pval] = corr(PH_Omegas_PD(H_inds, 28), H_UPDRS');
suptitle (strcat("Correlation = ", num2str(round(rho,2,'significant')), ", p-value = ", num2str(round(pval,2,'significant'))));

yyaxis left, plot(PH_Omegas_PD(H_inds, 28)), 
xlabel('Patient IDs')
xticks(1:6)
xticklabels({'4', '7', '8', '12', '13', '14'})
% xtext = "";
% for i = 1 : 6
%     xtext = strcat(xtext,",", num2str(H_inds(i)));
% end
% xticklabels({xtext});
ylabel('Phase of Omega of 28th mode')

yyaxis right, plot(H_UPDRS);
xlabel('Patient IDs')
ylabel("UPDRS: >= 31")

%-------------------------------------
[rho,pval] = corr(PH_Omegas_PD(L_inds, 59), L_UPDRS');
suptitle (strcat("Correlation = ", num2str(round(rho,2,'significant')), ", p-value = ", num2str(round(pval,2,'significant'))));

yyaxis left, plot(PH_Omegas_PD(L_inds, 1)), 
xlabel('Patient IDs')
xticks(1:6)
xticklabels({'3', '6', '9', '11', '15', '20'})
% xtext = "";
% for i = 1 : 6
%     xtext = strcat(xtext,",", num2str(H_inds(i)));
% end
% xticklabels({xtext});
ylabel('Phase of Omega of 1st mode')

yyaxis right, plot(L_UPDRS);
xlabel('Patient IDs')
ylabel("UPDRS: < 15")
%------------------------------------- LAMBDA-------------------------------
[rho,pval] = corr(PH_Lambdas_PD(H_inds, 15), H_UPDRS');
suptitle (strcat("Correlation = ", num2str(round(rho,2,'significant')), ", p-value = ", num2str(round(pval,2,'significant'))));

yyaxis left, plot(PH_Lambdas_PD(H_inds, 15)), 
xlabel('Patient IDs')
xticks(1:6)
xticklabels({'4', '7', '8', '12', '13', '14'})
% xtext = "";
% for i = 1 : 6
%     xtext = strcat(xtext,",", num2str(H_inds(i)));
% end
% xticklabels({xtext});
ylabel('Phase of Lambda of 15th mode')

yyaxis right, plot(H_UPDRS);
xlabel('Patient IDs')
ylabel("UPDRS: >= 31")

%-------------------------------------
[rho,pval] = corr(PH_Lambdas_PD(L_inds, 18), L_UPDRS');
suptitle (strcat("Correlation = ", num2str(round(rho,2,'significant')), ", p-value = ", num2str(round(pval,2,'significant'))));

yyaxis left, plot(PH_Lambdas_PD(L_inds, 1)), 
xlabel('Patient IDs')
xticks(1:6)
xticklabels({'3', '6', '9', '11', '15', '20'})
% xtext = "";
% for i = 1 : 6
%     xtext = strcat(xtext,",", num2str(H_inds(i)));
% end
% xticklabels({xtext});
ylabel('Phase of Lambda of 18th mode')

yyaxis right, plot(L_UPDRS);
xlabel('Patient IDs')
ylabel("UPDRS: < 15")
