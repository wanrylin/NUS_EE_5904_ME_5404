% EE5904 SVM Project

%%%%% read before run %%%%%
% please directly run the 1st (this) and 3rd section ( 2nd section is used for
% find the best parameter)

% task 3
clc
clear all
close all
load('train.mat');
load('test.mat');
load('eval.mat');

% standardization
mu = mean(train_data,2);
sigma = std(train_data,1,2); % normalized by the number of N
norm_train = (train_data - mu)./sigma;
norm_test = (test_data - mu)./sigma;
norm_eval = (eval_data - mu)./sigma;

% generate eval
[feature,N]=size(norm_train);
% eval_data = rand(feature, 600);
% eval_label = randi([1 2], [600 1]);
% eval_label = 2 * eval_label - 3;
% save('eval.mat', 'eval_data', 'eval_label');

%% 2 find the best hyperparameter for RBF-SVM
gamma = 1/feature .* [0.1,0.5,1,1.5,2];
C_values = [70,85,100,115,130,150];
iteration = 1;
for g = gamma
    % RBF kernel
    for i = 1:N
        for j = 1:i
            gram_matrix(i,j) = exp(-g * norm(norm_train(:,i) - norm_train(:,j)));
            H_matrix(i,j) = train_label(i) * train_label(j) * gram_matrix(i,j) ;
            gram_matrix(j,i) = gram_matrix(i,j);
            H_matrix(j,i) = H_matrix(i,j);
        end
    end

    % Mercer condition
    flag = mercer(H_matrix);
    % if not support Mercer condition, skip this gamma
    if flag == 1
        fprintf("g = %f is not admissible",g);
        continue
    end

    for C = C_values
        % compute alpha via quadprog
        % Set the training parameters
        [~,N]=size(norm_train);
        f = -1 * ones(N,1);
        A = [];
        B = [];
        Aeq = train_label';
        beq = 0;
        lb = zeros(N, 1);
        ub = ones(N, 1) * C;
        % set the conditions of quadprog
        x0 = [];
        opt = optimset('LargeScale', 'off', 'MaxIter', 1000);
        % Quadratic Programming
        alpha = quadprog(H_matrix,f,A,B,Aeq,beq,lb,ub,x0,opt);
        alpha(abs(alpha) < 1e-8) = 0;
        idx = find(alpha > 0 & alpha < C);
        rbf = [];
        for i = 1:length(idx)
            for j = 1:N
                rbf(i,j) = exp(-g * norm(norm_train(:,idx(i)) - norm_train(:,j)));
            end
        end
        w = rbf * (alpha .* train_label);
        b = mean(train_label(idx) - w);

        % training set accuracy
        rbf = [];
        for i=1:N
            for j = 1:N
                rbf(i,j) = exp(-g * norm(norm_train(:,i) - norm_train(:,j)));
            end
        end
        gx = rbf * (alpha .* train_label);
        train_pred = sign(gx + b);
        train_acc = sum(train_pred == train_label) / size(norm_train, 2);

        % test set accuracy
        rbf = [];
        for i=1:size(norm_test,2)
            for j = 1:N
                rbf(i,j) = exp(-g * norm(norm_test(:,i) - norm_train(:,j)));
            end
        end
        gx = rbf * (alpha .* train_label);
        test_pred = sign(gx + b);
        test_acc = sum(test_pred == test_label) / size(norm_test, 2);

        % record performance
        performance(iteration,:) = [train_acc,test_acc];
        fprintf("the performance of g = %f, C= %f is train acc = %f, test acc = %f \n",g,C,train_acc,test_acc )
        iteration = iteration + 1;
    end

end

%% 3 For evaluation
C_best = 100;
g_best = 1/feature;
Lable_str = ["non-spam","spam"];
Lable_str = categorical(Lable_str);

% RBF kernal
for i = 1:N
    for j = 1:i
        gram_matrix(i,j) = exp(-g_best * norm(norm_train(:,i) - norm_train(:,j)));
        H_matrix(i,j) = train_label(i) * train_label(j) * gram_matrix(i,j) ;
        gram_matrix(j,i) = gram_matrix(i,j);
        H_matrix(j,i) = H_matrix(i,j);
    end
end

% Mercer condition
flag = mercer(H_matrix);
% if not support Mercer condition, skip this gamma
if flag == 1
    fprintf("g = %f is not admissible",g_best);
end

% compute alpha via quadprog
% Set the training parameters
[~,N]=size(norm_train);
f = -1 * ones(N,1);
A = [];
B = [];
Aeq = train_label';
beq = 0;
lb = zeros(N, 1);
ub = ones(N, 1) * C_best;
% set the conditions of quadprog
x0 = [];
opt = optimset('LargeScale', 'off', 'MaxIter', 1000);
% Quadratic Programming
alpha = quadprog(H_matrix,f,A,B,Aeq,beq,lb,ub,x0,opt);
alpha(abs(alpha) < 1e-8) = 0;
idx = find(alpha > 0 & alpha < C_best);
rbf = [];
for i = 1:length(idx)
    for j = 1:N
        rbf(i,j) = exp(-g_best * norm(norm_train(:,idx(i)) - norm_train(:,j)));
    end
end
w = rbf * (alpha .* train_label);
b = mean(train_label(idx) - w);

% training set accuracy
rbf = [];
for i=1:N
    for j = 1:N
        rbf(i,j) = exp(-g_best * norm(norm_train(:,i) - norm_train(:,j)));
    end
end
gx = rbf * (alpha .* train_label);
train_pred = sign(gx + b);
train_acc = sum(train_pred == train_label) / size(norm_train, 2);
[cm,~] = confusionmat(train_label,train_pred);
figure();
title = ['Confusion Matrix for train set from RBF SVM with accuracy ',num2str(train_acc)];
confusionchart(cm, Lable_str, ...
    'Title',title)

% test set accuracy
rbf = [];
for i=1:size(norm_test,2)
    for j = 1:N
        rbf(i,j) = exp(-g_best * norm(norm_test(:,i) - norm_train(:,j)));
    end
end
gx = rbf * (alpha .* train_label);
test_pred = sign(gx + b);
test_acc = sum(test_pred == test_label) / size(norm_test, 2);
[cm,~] = confusionmat(test_label,test_pred);
figure();
title = ['Confusion Matrix for test set from RBF SVM with accuracy ',num2str(test_acc)];
confusionchart(cm, Lable_str, ...
    'Title',title);

% eval set accuracy
rbf = [];
for i=1:size(norm_eval,2)
    for j = 1:N
        rbf(i,j) = exp(-g_best * norm(norm_eval(:,i) - norm_train(:,j)));
    end
end
gx = rbf * (alpha .* train_label);
eval_pred = sign(gx + b);
eval_acc = sum(eval_pred == eval_label) / size(norm_eval, 2);
[cm,~] = confusionmat(eval_label,eval_pred);
figure();
title = ['Confusion Matrix for eval set from RBF SVM with accuracy ',num2str(eval_acc)];
confusionchart(cm, Lable_str, ...
    'Title',title);


function [flag] = mercer(matrix)
% check the mercer condition
eigenvalues = eig(matrix);
thre = 1e-4;
eigenvalues(abs(eigenvalues)<thre) = 0;
negative_values = eigenvalues < 0;

if (sum(negative_values) == 0)
    flag = 0;
else
    flag = 1;

end
end