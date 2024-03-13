close all
clear all
clc

%% load weights of NN
load('networks_weights/resweights_5_32.mat')
layers = length(W);

for ll = 1:4
    %% set P/S to zero
    if ll == 1
        vars = 'PSzero';
    elseif ll == 2
        vars = 'Szero';
    elseif ll == 3
        vars = 'Pzero';
    elseif ll == 4
        vars = 'PSnonzero';
    end
    
    %% LipSDP-GS
    [L,status] = LipschitzEstimationRes(W,vars);
    L_vec(ll) = L*norm(W{1})*norm(W{layers})
    
    info{ll} = status.info
    time(ll) = status.solvertime
end

%% MP bound
Ltriv = norm(W{1});
for ii = 2:layers-1
    Ltriv = Ltriv*(1+norm(W{ii}));
end
Ltriv = Ltriv*norm(W{layers})