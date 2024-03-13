function [L,diagnostics] = LipschitzEstimation_ResReLU_8(W,type)
n_l = 8;
nm = n_l-1;
nu = size(W{1},2);
nh = size(W{1},1);

rho = sdpvar;
lam = sdpvar(nm*nh,1);
T = diag(lam);
cons = [rho>=0,lam>=0];

ng = 2;
N = nh/2;

W1 = kron(eye(N),[-1 1; 1 -1]);
G = kron(eye(N),[-1 0; 0 1]);
H = kron(eye(N),[0 1; 0 1]);

Atilde = [W1*W{1}                      zeros(nh,7*nh);...
    W1*W{2}*H*W{1}               W1*W{2}*G               zeros(nh,6*nh);...
    W1*W{3}*H*W{2}*H*W{1}        W1*W{3}*H*W{2}*G        W1*W{3}*G zeros(nh,5*nh);...
    W1*W{4}*H*W{3}*H*W{2}*H*W{1} W1*W{4}*H*W{3}*H*W{2}*G W1*W{4}*H*W{3}*G     W1*W{4}*G zeros(nh,4*nh);...
    W1*W{5}*H*W{4}*H*W{3}*H*W{2}*H*W{1} W1*W{5}*H*W{4}*H*W{3}*H*W{2}*G W1*W{5}*H*W{4}*H*W{3}*G     W1*W{5}*H*W{4}*G W1*W{5}*G zeros(nh,3*nh);...
    W1*W{6}*H*W{5}*H*W{4}*H*W{3}*H*W{2}*H*W{1} W1*W{6}*H*W{5}*H*W{4}*H*W{3}*H*W{2}*G W1*W{6}*H*W{5}*H*W{4}*H*W{3}*G     W1*W{6}*H*W{5}*H*W{4}*G W1*W{6}*H*W{5}*G W1*W{6}*G zeros(nh,2*nh);...
    W1*W{7}*H*W{6}*H*W{5}*H*W{4}*H*W{3}*H*W{2}*H*W{1} W1*W{7}*H*W{6}*H*W{5}*H*W{4}*H*W{3}*H*W{2}*G W1*W{7}*H*W{6}*H*W{5}*H*W{4}*H*W{3}*G     W1*W{7}*H*W{6}*H*W{5}*H*W{4}*G W1*W{7}*H*W{6}*H*W{5}*G W1*W{7}*H*W{6}*G W1*W{7}*G zeros(nh,nh)];
Btilde = [zeros(7*nh,nu) eye(7*nh)];
Ctilde = [W{8}*H*W{7}*H*W{6}*H*W{5}*H*W{4}*H*W{3}*H*W{2}*H*W{1} W{8}*H*W{7}*H*W{6}*H*W{5}*H*W{4}*H*W{3}*H*W{2}*G W{8}*H*W{7}*H*W{6}*H*W{5}*H*W{4}*H*W{3}*G W{8}*H*W{7}*H*W{6}*H*W{5}*H*W{4}*G W{8}*H*W{7}*H*W{6}*H*W{5}*G W{8}*H*W{7}*H*W{6}*G W{8}*H*W{7}*G W{8}*G];

switch type
    case 'l2'
        
        D = -Ctilde'*Ctilde+[rho*eye(nu) zeros(nu,7*nh); ...
            zeros(7*nh,nu+7*nh)];
        
        diagnostics = optimize([[Atilde; Btilde]'*[zeros(7*nh,7*nh) -T; -T 2*T]*[Atilde; Btilde]+D>=0,cons],rho);
        L = sqrt(value(rho));
        
    case 'linfty'
        mu = sdpvar(nu,1);
        cons = [cons,mu>=0];
        
        Qtilde = blkdiag(diag(mu),zeros(7*nh,7*nh));
        D = [-sum(mu)+2*rho -Ctilde;...
            -Ctilde' Qtilde];
        
        diagnostics = optimize([[zeros(7*nh,1) Atilde; zeros(7*nh,1) Btilde]'...
            *[zeros(7*nh,7*nh) -T; -T 2*T]*[zeros(7*nh,1) Atilde; zeros(7*nh,1) Btilde]+D>=0,cons],rho);
        
        L = value(rho);
end


end