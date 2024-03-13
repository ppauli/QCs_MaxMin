function [L,diagnostics] = LipschitzEstimation_ResReLU_2(W,type)
n_layers = length(W);
nu = size(W{1},2);
nh = size(W{1},1);

rho = sdpvar;
lam = sdpvar(nh,1);
T = diag(lam);
cons = [rho>=0,lam>=0];

ng = 2;
N = nh/2;

W1 = kron(eye(N),[-1 1; 1 -1]);
G = kron(eye(N),[-1 0; 0 1]);
H = kron(eye(N),[0 1; 0 1]);

Atilde = [W1*W{1} zeros(nh,nh)];
Btilde = [zeros(nh,nu) eye(nh)];

switch type
    case 'l2'
        Ctilde = [W{2}*H*W{1} W{2}*G];
        D = -Ctilde'*Ctilde+[rho*eye(nu) zeros(nu,nh); ...
            zeros(nh,nu+nh)];
        
        diagnostics = optimize([[Atilde; Btilde]'*[zeros(nh,nh) -T; -T 2*T]*[Atilde; Btilde]+D>=0,cons],rho);
        L = sqrt(value(rho));
        
    case 'linfty'
        mu = sdpvar(nu,1);
        cons = [cons,mu>=0];
        
        Ctilde = [W{2}*H*W{1} W{2}*G];
        Qtilde = blkdiag(diag(mu),zeros(nh,nh));
        D = [-sum(mu)+2*rho -Ctilde;...
            -Ctilde' Qtilde];
        
        diagnostics = optimize([[zeros(nh,1) Atilde; zeros(nh,1) Btilde]'...
            *[zeros(nh,nh) -T; -T 2*T]*[zeros(nh,1) Atilde; zeros(nh,1) Btilde]+D>=0,cons],rho);
        
        L = value(rho);
end


end