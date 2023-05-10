% EE5904 SVM Project

% task 2
clc
clear all
close all
load('train.mat');
load('test.mat');
% read the result of the task 1
load('task_1.mat')

%% (i) show the performance of the hard-margin SVM with the linear kernel
% for linear kernel
p_linear = []; % linear kernal performance
% train set
train_pred = sign(norm_train' * w_linear + b_linear);
train_acc = sum(train_pred == train_label) / size(norm_train, 2);
% test set
test_pred = sign(norm_test' * w_linear + b_linear);
test_acc = sum(test_pred == test_label) / size(norm_test, 2);
p_linear = [train_acc,test_acc];
fprintf("Train accuracy: %f;  Test accuracy: %f \n",train_acc,test_acc);

%% (ii) show the performance of the hard-margin SVM with the polynomial kernel
p_hard = []; % polynomial kernal performance
i = 1;
for p = 2:5
    % train set
    gx = alpha_hard{i} .* train_label .* (norm_train' * norm_train + 1) .^ p;
    gx = sum(gx,1)';
    train_pred = sign(gx + b_hard{i});
    train_acc = sum(train_pred == train_label) / size(norm_train, 2);
    % test set
    gx = alpha_hard{i} .* train_label .* (norm_train' * norm_test + 1) .^ p;
    gx = sum(gx,1)';
    test_pred = sign(gx + b_hard{i});
    test_acc = sum(test_pred == test_label) / size(norm_test, 2);
    p_hard(i,:) = [train_acc,test_acc];
    fprintf("Train accuracy: %f;  Test accuracy: %f\n",train_acc,test_acc);
    i = i + 1;
end



%% (ii) show the performance of the hard-margin SVM with the polynomial kernel
p_poly = []; % polynomial kernal performance
i = 1;
C_values = [0.1,0.6,1.1,2.1];
for p = 1:5
    for C = C_values
        % train set
        train_pred = sign(((norm_train' * norm_train + 1) .^ p) * (alpha_poly{i} .* train_label) + b_poly{i});
        train_acc = sum(train_pred == train_label) / size(norm_train, 2);
        % test set
        test_pred = sign(((norm_test' * norm_train + 1) .^ p) * (alpha_poly{i} .* train_label) + b_poly{i});
        test_acc = sum(test_pred == test_label) / size(norm_test, 2);
        p_poly(i,:) = [train_acc,test_acc];
        fprintf("Train accuracy: %f;  Test accuracy: %f\n",train_acc,test_acc);
        i = i + 1;
    end
end

