function [L,diagnostics] = LipschitzEstimationRes(W,vars)
n_layers = length(W);
nx = size(W{2},1);

rho = sdpvar;
cons = [rho>=0];

Atilde = [];
for ii = 2:n_layers-1
    tmp = zeros((ii-2)*nx,nx);
    for jj = ii:n_layers-1
        tmp = [tmp; W{jj}];
    end
    Atilde = [Atilde, tmp];
end
Atilde = [Atilde, zeros((n_layers-2)*nx,nx)];
Btilde = [zeros(nx*(n_layers-2),nx) eye((n_layers-2)*nx)];

G = [];
for ii = 1:n_layers-1
    tmp = [];
    for jj = 1:n_layers-1
        tmp = [tmp eye(nx)];
    end
    G = [G;tmp];
end

R = [rho*eye(nx) zeros(nx,(n_layers-2)*nx); ...
    zeros((n_layers-2)*nx,(n_layers-1)*nx)];

G = G-R;

N = (n_layers-2)*nx/2;
ng = 2;

[T,lam] = get_T(N,ng);

switch vars
    case 'PSnonzero'
    S = get_P(N,ng);
    P = get_P(N,ng);
    cons = [cons, lam>=0, [Atilde; Btilde]'*[T-2*S P+S; P+S -T-2*P]*[Atilde; Btilde]+G<=0];
    case 'Pzero'
    S = get_P(N,ng);
    cons = [cons, lam>=0, [Atilde; Btilde]'*[T-2*S S; S -T]*[Atilde; Btilde]+G<=0];
    case 'Szero'
    P = get_P(N,ng);
    cons = [cons, lam>=0, [Atilde; Btilde]'*[T P;P -T-2*P]*[Atilde; Btilde]+G<=0];
    case 'PSzero'
    cons = [cons, lam>=0, [Atilde; Btilde]'*[T zeros(size(T)); zeros(size(T)) -T]*[Atilde; Btilde]+G<=0];
end

ops = sdpsettings('solver','mosek','verbose',1,'debug',1,'dualize',0);

diagnostics = optimize(cons,rho,ops);

L = sqrt(value(rho));

end

function [con,Q] = get_con_firstlast(W,Tm,firstlast)

[nout,~] = size(W);

switch firstlast
    case 'first'
        Q = sdpvar(nout,nout);
        con = [Tm-W'*Q*W>=0];
    case 'last'
        con = [Tm-W'*W>=0];
        Q = [];    
end
end

function [T,lam1] = get_T(N,ng)
    lam1 = sdpvar(N,1);
    lam2 = sdpvar(N,1);
    T = [];
    for jj = 1:N
        T = blkdiag(T,lam1(jj)*eye(ng)+lam2(jj)*ones(ng,ng));
    end
end

function P = get_P(N,ng)
    lam2 = sdpvar(N,1);
    P = [];
    for jj = 1:N
        P = blkdiag(P,lam2(jj)*ones(ng,ng));
    end
end