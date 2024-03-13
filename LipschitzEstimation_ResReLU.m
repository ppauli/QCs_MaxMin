function [L,diagnostics] = LipschitzEstimation_ResReLU(W,type)
n_layers = length(W);
nx = size(W{1},2);

rho = sdpvar;
cons = [rho>=0];

switch type
    case 'l2'
        tau = 0;
        Qm = rho*speye(nx);
    case 'linfty'
        tau = sdpvar(1,nx);
        Qm = diag(tau);
        cons = [cons, tau>=0];
end

for ii = 1:n_layers
    if ii == n_layers
        con = get_con(W{ii},Qm,1,type,tau,rho);
        cons = [cons,con];
    else
        [con,Qm] = get_con(W{ii},Qm,0,[],[]);
        cons = [cons,con];
    end
end

ops = sdpsettings('solver','mosek','verbose',1,'debug',1,'dualize',0);

diagnostics = optimize(cons,rho,ops);
switch type
    case 'l2'
        L = sqrt(value(rho));
    case 'linfty'
        L = value(rho);
end
end

function [con,Q] = get_con(W,Qm,last,type,tau,rho)

[nout,~] = size(W);

if last == 1
    switch type
        case 'l2'
            con = [Qm-W'*W>=0];
            T = [];
        case 'linfty'
            con = [[Qm W'; W 2*rho-sum(tau)]>=0];
            T = [];            
    end
else
    ng = 2;
    N = nout/2;
    lam = sdpvar(nout,1);
    T = diag(lam);
    Q = sdpvar(nout,nout);
    
    W1 = kron(eye(N),[-1 1; 1 -1]);
    G = kron(eye(N),[-1 0; 0 1]);
    H = kron(eye(N),[0 1; 0 1]);
    
    con = [[Qm-W'*H'*Q*H*W W'*W1'*T+W'*H'*Q*G; T*W1*W+G'*Q*H*W 2*T-G'*Q*G]>=0,lam>=0];
end
end