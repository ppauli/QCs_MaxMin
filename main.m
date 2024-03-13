close all
clear all
clc

% To run this code, you require YALMIP and the solver Mosek

%% load weights of NN

str_net_vec = {...
    'weights_l2_2fc_16.mat','weights_l2_2fc_32.mat',...
    'weights_l2_2fc_64.mat','weights_l2_2fc_128.mat',...
    'weights_l2_5fc_32.mat','weights_l2_5fc_64.mat',...
    'weights_l2_8fc_32.mat','weights_l2_8fc_64.mat',...
    'weights_l2_8fc_128.mat','weights_l2_8fc_256.mat',...
    'weights_l2_18fc_32.mat','weights_l2_18fc_64.mat',...
    'weights_l2_18fc_128.mat'
    };
for kk = 1:length(str_net_vec)
str_net = str_net_vec{kk}
str = ['networks_weights/' str_net];

clear W
load(str)
layers = length(W);

clear L_vec L_ResReLU_vec L_SR_vec Ltriv_vec info time info_ResReLU ...
    time_ResReLU info_SR time_SR

for ll = 1:2
    ll
    if ll == 1
        type = 'l2';
    else
        type = 'linfty';
    end
    switch type
        case 'linfty'
            W{layers} = W{layers}(9,:); % to consider output for label 8
    end
    
    %% LipSDP-GS
    disp(['Starting LipSDP-NSR for ' str_net ' for ' type])
    [L,status] = LipschitzEstimation(W,type);
    %disp(['Starting LipSDP for residual ReLU for ' str_net ' for ' type])
    %[L_ResReLU,status_ResReLU] = LipschitzEstimation_ResReLU_8(W,type); !! use the correct hard coded version !!
    
    %% MP bound
    Ltriv = 1;
    
    for ii = 1:length(W)
        switch type
            case 'l2'
                Ltriv = norm(W{ii})*Ltriv;
            case 'linfty'
                Ltriv = norm(W{ii}',1)*Ltriv;
        end
    end
    
    %% collect results
    L_vec(ll) = L
    %L_ResReLU_vec(ll) = L_ResReLU
    Ltriv_vec(ll) = Ltriv
    
    info{ll} = status.info
    time(ll) = status.solvertime
    %info_ResReLU{ll} = status_ResReLU.info
    %time_ResReLU(ll) = status_ResReLU.solvertime

end
%save(['results\res_' str_net],'L_vec','L_ResReLU_vec','Ltriv_vec','info','time','info_ResReLU','time_ResReLU')
save(['results\res_' str_net],'L_vec','Ltriv_vec','info','time')
end