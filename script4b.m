%% Forward vs. Reverse Inference Simulation

clear; clc; close all;
rng(42);

nSubjects = 100;
% Task labels: A = 'conflict', B = 'emotion'
task_labels = [repmat("A",nSubjects/2,1); repmat("B",nSubjects/2,1)];
task_codes = [ones(nSubjects/2,1); zeros(nSubjects/2,1)];

% Simulate activation in Region R
% Conflict task (A) activates R more strongly
act_R = zeros(nSubjects,1);
act_R(task_codes==1) = randn(nSubjects/2,1)*0.5 + 1.5;  % Task A
act_R(task_codes==0) = randn(nSubjects/2,1)*0.5 + 1.0;  % Task B

%% Forward Inference: Task → Activation
% Does Task A activate Region R more than Task B?
[h,p,ci,stats] = ttest2(act_R(task_codes==1), act_R(task_codes==0));
fprintf('Forward Inference:\n');
fprintf('t(%.0f) = %.2f, p = %.4f\n', stats.df, stats.tstat, p);

%% Reverse Inference: Activation → Task
% If Region R is active, what's the probability it's Task A?
threshold = 1.2;
nA_given_R = sum(task_codes==1 & act_R > threshold);
nR = sum(act_R > threshold);
p_A_given_R = nA_given_R / nR;

fprintf('Reverse Inference:\n');
fprintf('P(Task A | Activation > %.1f) = %.2f\n', threshold, p_A_given_R);

%% Plot for Forward vs Reverse Inference
figure('Color','w'); hold on;

% Plot histograms with transparency to show overlap
hA = histogram(act_R(task_codes==1), 'BinWidth',0.2, ...
    'FaceColor',[0.2 0.6 1], 'FaceAlpha',0.6, 'EdgeColor','none', ...
    'DisplayName','Task A (Conflict)');
hB = histogram(act_R(task_codes==0), 'BinWidth',0.2, ...
    'FaceColor',[1 0.5 0.2], 'FaceAlpha',0.6, 'EdgeColor','none', ...
    'DisplayName','Task B (Emotion)');

% Threshold line
threshold = 1.2;
xline(threshold, 'k--', 'LineWidth', 2, ...
    'DisplayName', sprintf('Activation Threshold = %.1f', threshold));

xlabel('Activation in Region R');
ylabel('Number of Subjects');
title('Forward vs. Reverse Inference');
legend('Location','northwest');
grid on;
