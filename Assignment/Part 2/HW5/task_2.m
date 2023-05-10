% EE5904 RL project



clc
clear all
close all
% load("task1.mat");
load("qeval.mat");
rng(5904); % random seed


%% Initialization
[num_state,num_action] = size(reward);
gamma = 0.7;
mode_type = 6; % alpha_k type
max_trial = 3000;
q_threshold = 0.05;
iteration = length(mode_type) * length(gamma);

%% Start run
num_run = zeros(1, iteration);
execution_time = zeros(1, iteration);
best_reward = zeros(1, iteration);
opt_policy = cell(1, iteration);
opt_Q = cell(1, iteration);
opt_trace = cell(1, iteration);
situation = 1;
for mode = mode_type
    for g = gamma
        reach_times = 0;
        max_reward = 0;
        run_times = [];
        optimal_trace = [];
        optimal_Q = [];
        optimal_policy = [];
        for run = 1:10
            % print current status
            k = 1;
            [f_name,~] = Learningrate(k,mode);
            fprintf('%s  : %d times run\n',f_name,run);
            % execution time clock
            tic;
            % initialize parameters
            trial = 1;
            start_state = 1;
            end_state = 100;
            Q = zeros(num_state,num_action);
            converge = false;

            % start trial
            while trial <= max_trial && ~converge
                % first attempt
                k = 1;
                s_k = start_state;
                Q_old = Q;

                % start moving
                while s_k ~= end_state
                    % exploration or exploitation
                    [~,epsilon_k] = Learningrate(k,mode);
                    if epsilon_k < 0.005
                        break
                    end
                    % take a movement
                    a_k = Action(epsilon_k,Q(s_k,:),reward(s_k,:));
                    % update Q function
                    [Q_new,next] = Qupdate(Q,a_k,s_k,epsilon_k,reward,g);
                    Q(s_k, a_k) = Q_new;
                    s_k = s_k + next;
                    % update k
                    k = k + 1;
                end
                % update trial and flag
                trial = trial + 1;
                if max(abs(Q_old - Q)) < q_threshold
                    converge = true;
                end
            end

            elapsedTime = toc;
            % calculate the coordinates of optimal policy
            [reach_flag,policy_reward,trace,policy] = resultcheck(Q,reward,start_state,end_state,g);
            total_reward = sum(policy_reward,'all');
            if reach_flag == true
                if total_reward > max_reward
                    max_reward = total_reward;
                    optimal_trace = trace;
                    optimal_Q = Q;
                    optimal_policy = policy;
                end
                % update reach times
                reach_times = reach_times + 1;
                run_times = [run_times, elapsedTime];
            end
        end
        if reach_times == 0
            fprintf('mode: %d, gamma: %f  fail to reach the terminal\n',mode,g);
        elseif reach_times ~= 0
            fprintf('mode: %d, gamma: %f  successfully reach the terminal\n',mode,g);
            num_run(situation) = reach_times;
            execution_time(situation) = mean(run_times);
            opt_trace{situation} = optimal_trace;
            opt_policy{situation} = optimal_policy;
            opt_Q{situation} = optimal_Q;
            best_reward(situation) = max_reward;
        end
        situation = situation + 1;
    end
end

%% Draw max reward policy
i = 1;
for mode = mode_type
    for g = gamma
        if isempty(opt_policy{i})
            i = i + 1;
            continue
        end
        disp(i)
        draw_path_policy(best_reward(i), opt_trace{i}, opt_Q{i}, g, mode, execution_time(i))
        i = i + 1;
    end
end

%% conclusion
i = 1;
for mode = mode_type
    for g = gamma
        k = 1;
        [f_name,~] = Learningrate(k,mode);
        disp([f_name,' \gamma',num2str(g),' model'])
        if isempty(opt_policy{i})
            fprintf('success runs: 0, execution time: N/A\n');
        else
            fprintf('success runs: %d, execution time: %f\n',num_run(i),execution_time(i));
        end
        i = i + 1;
    end
end

% learning_rate_plot(mode_type);


%% Functions
function [f_name,rate] = Learningrate(k,mode)
switch mode
    case 1
        rate = 100 ./ (100 + k);
        f_name = 'Rate = ^{100}/_{100 + k}';
    case 2
        rate = (1 + 5 .* log(k)) ./ k;
        f_name = 'Rate = ^{1 + 5log(k)}/_{k}';
    case 3
        rate = 100 ./ (100 + sqrt(k));
        f_name = 'Rate = ^{100}/_{100 + k^{0.5}}';
    case 4
        rate = (1 + 10 .* log(k)) ./ k;
        f_name = 'Rate = ^{1 + 10log(k)}/_{k}';
    case 5
        rate = exp(-0.001.*k);
        f_name = 'Rate = exp(-0.001k)';
    case 6
        rate =  1 ./ k .^ 0.1;
        f_name = 'Rate = ^{1}/_{k^0.1}';
    otherwise
        error('out of existing mode');
end
rate(rate>1) = 1;
end

function act = Action(epsilon_k,Q_sk,alternative)
valid_idx=find(alternative ~= -1);
if any(Q_sk) % not all 0
    random = rand;

    % Exploitation
    if random > epsilon_k
        [~,max_idx] = max(Q_sk(valid_idx));
        act = valid_idx(max_idx);
        % Exploration
    elseif random <= epsilon_k
        other_idx = find(Q_sk(valid_idx) ~= max(Q_sk(valid_idx)));
        rand_idx = randperm(length(other_idx),1);
        act=valid_idx(other_idx(rand_idx));
    end
    %     return;
    % elseif ~any(Q_sk) % all 0
else
    rand_idx = randperm(length(valid_idx),1);
    act = valid_idx(rand_idx);
end
end

function [Q_new,state] = Qupdate(Q,a_k,s_k,epsilon_k,reward,g)
% next state
state = 10 ^ (mod(a_k + 1, 2)) * (-1) ^ (floor(a_k / 2) + 1);
% update Q value
Q_new = Q(s_k, a_k) + epsilon_k * (reward(s_k, a_k) + g * max(Q(s_k + state, :)) - Q(s_k, a_k));
% update s_k
s_k = s_k + state;
end

function [flag,policy_reward,trace,policy] = resultcheck(Q,reward,start_state,end_state,g)
[~,policy]=max(Q,[],2); % find max value for each state
state = start_state;
step = 1;
policy_reward = zeros(10,10);
trace = []; % record the trace of the robot
while state ~= end_state && policy_reward(mod(state-1,10)+1,floor(state/10)+1) == 0
    trace = [trace, policy(state)]; % record trace
    policy_reward(mod(state - 1,10) + 1,floor(state/10)+1) = g ^(step - 1) * reward(state,policy(state)); % record reward
    state = state + (10 ^ (mod(policy(state) + 1, 2)) * (-1) ^ (floor(policy(state) / 2) + 1)); % next state
    step = step + 1;
end
if state == 100
    trace = [trace, state];
    flag = true;
    fprintf('Success!\n')
else
    flag = false;
    fprintf('Fail!\n')
end
end

function draw_path_policy(max_reward, policy, Q,g,mode,time)
k = 1;
[f_name,~] = Learningrate(k,mode);
n_path = length(policy);
state = 1;
direction_blue = ['^b';'>b';'vb';'<b'];
direction_red = ['^r';'>r';'vr';'<r'];

figure();
hold on
plot(9.5, 9.5, '*r', 'LineWidth',2)
axis([0 10 0 10])
title({f_name;['\gamma',num2str(g),' Execution reward = ',num2str(max_reward), ' time = ', num2str(time)]},'FontSize',12)
grid on
set(gca,'YDir','reverse')   % grid from top left corner
for i = 1 : n_path - 1
    x = floor((state  - 1) / 10) + 0.5;
    y = mod(state - 1, 10) + 0.5;
    a_k = policy(i);
    plot(x, y, direction_blue(a_k, :), 'LineWidth',2)
    next = 10 ^ (mod(a_k + 1, 2))  * (-1) ^ (floor(a_k / 2) + 1);
    state = state + next;
end
hold off
% opt policy
[~, opt_act]=max(Q,[],2);
n_act = length(opt_act);
figure();
hold on
plot(9.5, 9.5, '*b', 'LineWidth',2)
axis([0 10 0 10])
title({f_name;['\gamma',num2str(g),' Optimal policy reward = ',num2str(max_reward), ' time = ', num2str(time)]},'FontSize',12)
grid on
set(gca,'YDir','reverse')
for i = 1 : n_act
    x = floor((i  - 1) / 10) + 0.5;
    y = mod(i - 1, 10) + 0.5;
    plot(x, y, direction_red(opt_act(i), :),'LineWidth',2)
end
hold off
end

function learning_rate_plot(mode_type)
figure();
k = 0:1:2000;
hold on
grid on
func = cell(1,length(mode_type));
for mode = mode_type
    [f_name,rate] = Learningrate(k,mode);
    plot(k,rate,'LineWidth',2)
    func{mode} = f_name;
end

legend(func,'Location','northeast','FontSize',11)
title('\epsilon_k, \alpha_k decay against Time step','FontSize',14)
xlabel('Time step k','FontSize',12);
ylabel('Rate decay','FontSize',12);
axis([0,2000,0,1])
end

