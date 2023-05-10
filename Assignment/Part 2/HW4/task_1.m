% EE5904 SVM Project


% task 1
clc
clear all
close all
load('train.mat');
load('test.mat')

% standardization
mu = mean(train_data,2);
sigma = std(train_data,1,2); % normalized by the number of N
norm_train = (train_data - mu)./sigma;
norm_test = (test_data - mu)./sigma;


%% (i)A hard-margin SVM with the linear kernel
C = 1e6;
gram_matrix = norm_train' * norm_train;
mercer(gram_matrix) % check mercer condition
H_linear = train_label * train_label' .*gram_matrix;
% compute alpha via quadprog
% Set the training parameters
[~,N]=size(norm_train);
f = -1 * ones(N,1);
A = [];
b = [];
Aeq = train_label';
beq = 0;
lb = zeros(N, 1);
ub = ones(N, 1) * C;
% set the conditions of quadprog
x0 = [];
opt = optimset('LargeScale', 'off', 'MaxIter', 1000);
% Quadratic Programming
alpha = quadprog(H_linear,f,A,b,Aeq,beq,lb,ub,x0,opt);
idx = find(alpha > 1e-4);
% Calculate disciminant parameters
w_linear = sum(alpha' .* train_label' .* norm_train,2);
b_linear =mean(1./train_label(idx) - norm_train(:,idx)'*w_linear);

fprintf('hard-margin SVM with the linear kernel done');

%% (ii) A hard-margin SVM with a polynomial kernel
C = 1e6;
i = 1;
for p = 2:5
    % Check Mercer condition
    gram_matrix = (norm_train' * norm_train+1).^p;
    mercer(gram_matrix);
    H_poly = train_label * train_label' .*gram_matrix;
    % compute alpha via quadprog
    % Set the training parameters
    [~,N]=size(norm_train);
    f = -1 * ones(N,1);
    A = [];
    b = [];
    Aeq = train_label';
    beq = 0;
    lb = zeros(N, 1);
    ub = ones(N, 1) * C;
    % set the conditions of quadprog
    x0 = [];
    opt = optimset('LargeScale', 'off', 'MaxIter', 1000);
    % Quadratic Programming
    alpha = quadprog(H_poly,f,A,b,Aeq,beq,lb,ub,x0,opt);
    alpha(abs(alpha) < 1e-8) = 0;
    idx = find(alpha > 0 && alpha < C);
    % Calculate disciminant parameters
    alpha_hard{i} = alpha;
    w_hard{i} = ((norm_train(:,idx)' * norm_train + 1) .^ p) * (alpha .* train_label);
    b_hard{i} = mean(train_label(idx) - w_hard{i});
    i = i + 1;
    fprintf('hard-margin SVM with a polynomial kernel %d done',i)
end

%% A soft-margin SVM with a polynomial kernel

i = 1;
C_values = [0.1,0.6,1.1,2.1];

for p = 1:5
    % Check Mercer condition
    gram_matrix = (norm_train' * norm_train+1).^p;
    mercer(gram_matrix);
    H_poly = train_label * train_label' .*gram_matrix;
    for C = C_values
        % compute alpha via quadprog
        % Set the training parameters
        [~,N]=size(norm_train);
        f = -1 * ones(N,1);
        A = [];
        b = [];
        Aeq = train_label';
        beq = 0;
        lb = zeros(N, 1);
        ub = ones(N, 1) * C;
        % set the conditions of quadprog
        x0 = [];
        opt = optimset('LargeScale', 'off', 'MaxIter', 1000);
        % Quadratic Programming
        alpha = quadprog(H_poly,f,A,b,Aeq,beq,lb,ub,x0,opt);
        alpha(abs(alpha) < 1e-8) = 0;
        idx = find(alpha > 0 && alpha < C);
        % Calculate disciminant parameters
        alpha_poly{i} = alpha;
        w_poly{i} = ((norm_train(:,idx)' * norm_train + 1) .^ p) * (alpha .* train_label);
        b_poly{i} = mean(train_label(idx) - w_poly{i});
        i = i + 1;
        fprintf('soft-margin SVM with a polynomial kernel C = %f, p = %d done',C,p)
    end
end

%% save the result of w and b
save('task_1.mat')




function mercer(matrix)
% check the mercer condition
eigenvalues = eig(matrix);
thre = 1e-4;
eigenvalues(abs(eigenvalues)<thre) = 0;
negative_values = eigenvalues < 0;

if (sum(negative_values) == 0)
    disp('Mercer condition passed!');
else
    disp('This kernel candidate is not admissible');

end
end