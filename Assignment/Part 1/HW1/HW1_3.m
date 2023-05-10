clc;
clear;
close all;

%% Q3
% ground truth
AND = [ 0 0 1 1 ; 0 1 0 1 ; 0 0 0 1];
OR = [ 0 0 1 1 ; 0 1 0 1 ; 0 1 1 1];
COMPLEMENT = [ 0 1 ; 1 0];
NAND = [ 0 0 1 1 ; 0 1 0 1 ; 1 1 1 0];
XOR = [ 0 1 0 1 ; 0 0 1 1 ; 0 1 1 0];

% learning parameter setting
gate = XOR;           %the logic gate
rate = 1;           %learning rate
[dim, num_input] = size(gate);
loop = 1;
error = zeros(1,num_input);

% % off-line calculation
% w_and = [-1.5,1,1];
% w_or = [-0.5,1,1];
% w_complement = [0.5,1];
% w_nand = [1.5,-1,-1];
% w = w_and;
% 
% figure;
% hold on;
% axis([-0.5,1.5,-0.5,1.5])
% for i = 1:num_input
%     if gate(end,i) == 1
%         plot(gate(1,i),gate(2,i),'bx');
%     else
%         plot(gate(1,i),gate(2,i),'ro');
%     end
% end
% x = linspace(-1,2,100);
% k = -w(end,2)/w(end,3);
% b = -w(end,1)/w(end,3);
% y = k * x + b;
% plot(x, y,'k')
% grid on
% hold off

% learning operation
w = rand(1,dim);
while true
    for i = 1 : num_input
        y = (w(loop,:) * [1;gate(1:dim-1 , i)]) > 0;
        error(1,i) = gate(dim,i) - y;
        if error(1,i) ~= 0
            w(loop+1,:) = w(loop,:) + (rate*error(1,i)*[1;gate(1:dim-1 , i)])';
            loop = loop + 1;
        end
    end
    if all(error == 0)
        break
    elseif loop > 1000
        break
    end
end


% plot
if dim == 2
    figure;
    hold on;
    xlabel("Iterations");
    ylabel("Weight values");
    x = 0:size(w,1)-1;
    plot(x,w(:,1),'-ro');
    plot(x,w(:,2),'-mx');
    legend({'w0','w1'});
    grid on
    hold off

    figure;
    hold on;
    axis([-0.5,1.5,-0.5,1.5])
    for i = 1:num_input
        if gate(end,i) == 1
            plot(0,gate(1,i),'bx');
        else
            plot(0,gate(1,i),'ro');
        end
    end
    x = linspace(-1,2,100);
    k = w(end,2);
    b = w(end,1);
    y = k * x + b;
    plot(x, y,'k')
    grid on
    hold off
end

if dim == 3
    figure;
    hold on;
    xlabel("Iterations");
    ylabel("Weight values");
    x = 0:size(w,1)-1;
    plot(x,w(:,1),'-ro');
    plot(x,w(:,2),'-kx');
    plot(x,w(:,3),'-b+');
    legend({'w0','w1','w2'});
    grid on
    hold off

    figure;
    hold on;
    axis([-0.5,1.5,-0.5,1.5])
    for i = 1:num_input
        if gate(end,i) == 1
            plot(gate(1,i),gate(2,i),'bx');
        else
            plot(gate(1,i),gate(2,i),'ro');
        end
    end
    x = linspace(-1,2,100);
    k = -w(end,2)/w(end,3);
    b = -w(end,1)/w(end,3);
    y = k * x + b;
    plot(x, y,'k')
    grid on
    hold off
end




