%% Q4a
clc;
clear all;
close all;

points = [0,0.5;0.8,1;1.6,4;3,5;4,6;5,9];
x = points(:,1);
y = points(:,2);
X = [ones(6,1),x];
w= (inv(X'*X)*X'*y)';

k = w(1,2);
b = w(1,1);

a = linspace(-1,6,100);
y = k * a + b;

hold on
plot(a, y,'k')
scatter(points(:,1), points(:,2),'x');
legend("LLS");
xlabel('x')
ylabel('y')
title("Linear Least Squares fitting")
grid;
hold off

%% Q4b
clear all;
close all;

points = [0,0.5;0.8,1;1.6,4;3,5;4,6;5,9];
x = points(:,1);
y = points(:,2);
X = [ones(6,1),x];
num_input = length(points);
weights = rand(1,2);            % initial weight is chosen randomly
rate = 0.1;
error_sum = zeros(100,1);

for i = 1:100
    for j = 1:6
        error = y(j) - weights(i,:)*X(j,:)';
        error_sum(i,1) = error^2/2 + error_sum(i,1);
        weights(i,:) = weights(i,:) + rate*error*X(j,:); 
    end
    weights(i+1,:) = weights(i,:);
end

figure
plot(1:100,error_sum,'k');
xlabel('Iterations')
ylabel('Err')
title("LMS fitting error")
grid on;

figure
hold on
plot(0:100, weights(:,1), 'k');
plot(0:100, weights(:,2), 'm');
legend("b", "w");
xlabel('Number of Iterations')
ylabel('Weight values')
title("Trajectories of the weights")
hold off
grid on;

k = weights(end,2);
b = weights(end,1);
a = linspace(-1,6,100);
y = k * a + b;
figure
hold on
plot(a, y,'k')
scatter(points(:,1), points(:,2),'x');
legend("LMS");
xlabel('x')
ylabel('y')
title("Least Mean Squares fitting")
grid;
hold off
