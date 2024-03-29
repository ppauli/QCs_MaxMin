function [L,diagnostics] = LipschitzEstimation(W,type)
n_layers = length(W);
nx = size(W{1},2);

rho = sdpvar;
cons = [rho>=0];

switch type
    case 'l2'
        tau = 0;
        Tm = rho*speye(nx);
    case 'linfty'
        tau = sdpvar(1,nx);
        Tm = diag(tau);
        cons = [cons, tau>=0];
end

for ii = 1:n_layers
    if ii == n_layers
        con = get_con(W{ii},Tm,1,type,tau,rho);
        cons = [cons,con];
    else
        [con,Tm] = get_con(W{ii},Tm,0,[],[]);
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

function [con,T] = get_con(W,Tm,last,type,tau,rho)

[nout,~] = size(W);

if last == 1
    switch type
        case 'l2'
            con = [Tm-W'*W>=0];
            T = [];
        case 'linfty'
            con = [[Tm W'; W 2*rho-sum(tau)]>=0];
            T = [];            
    end
else
    ng = 2;
    N = nout/2;
    lam1 = sdpvar(N,1);
    lam2 = sdpvar(N,1);
    T = [];
    for jj = 1:N
        T = blkdiag(T,lam1(jj)*eye(ng)+lam2(jj)*ones(ng,ng));
    end
    con = [Tm-W'*T*W>=0,lam1>=0];
end
end