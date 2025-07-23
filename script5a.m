%% 1. Classification Task: Decoding Color and Shape
%%% Goal: Use multivariate classifiers (LDA and SVM) to decode the stimulus color (red vs green) and shape (circle vs square) from simulated beta patterns.

%% Setup
rng(25);  % For reproducibility

% Define parameters
nTrials = 200;     % Number of trials
nVoxels = 100;     % Number of voxels per trial
nColorVoxels = 20; % Voxels coding color
nShapeVoxels = 20; % Voxels coding shape

%% Create label vectors
colors = repmat([1;2], nTrials/2,1);    % 1=red, 2=green
shapes = repmat([1;2], nTrials/2,1);    % 1=circle, 2=square

Ycolor = categorical(colors);
Yshape = categorical(shapes);

%% Simulate voxel patterns
% Random noise base
X = randn(nTrials, nVoxels);

% Add pattern differences for color
X(:,1:nColorVoxels) = X(:,1:nColorVoxels) + (colors - 1.5);  % shift -0.5 / +0.5

% Add pattern differences for shape
X(:,21:20+nShapeVoxels) = X(:,21:20+nShapeVoxels) + (shapes - 1.5);

%% Classification with LDA for Color
cv = cvpartition(Ycolor, 'KFold', 5); % 5-fold CV; defines a random partition on a data set.
opts = struct('CVPartition',cv);

ldaModel = fitcdiscr(...
    X, Ycolor, ...
    'DiscrimType', 'pseudoLinear', ...
    'CVPartition', cv);

acc_color = 1 - kfoldLoss(ldaModel);
fprintf('Color decoding accuracy (LDA): %.2f%%\n', acc_color*100);

%% Classification with SVM for Shape
svmModel = fitcsvm(X, Yshape, 'KernelFunction','linear', ...
    'Standardize',true, 'CVPartition',cv);
acc_shape = 1 - kfoldLoss(svmModel);
fprintf('Shape decoding accuracy (SVM): %.2f%%\n', acc_shape*100);

%% Feature Importance (LDA)
% Inspect mean activation differences between classes
% Extract trained models from CV folds
trainedModels = ldaModel.Trained;
nFolds = numel(trainedModels);
meanDiff = zeros(nFolds, nVoxels);

% Compute mean differences for each fold
for i = 1:nFolds
    meanDiff(i,:) = abs(diff(trainedModels{i}.Mu));
end

% Average across folds
meanDiffAvg = mean(meanDiff, 1);

% Plot feature importance
figure;
bar(meanDiffAvg); hold on;
xline(nColorVoxels, 'r--', 'Color boundary');
xline(20 + nShapeVoxels, 'b--', 'Shape boundary');
title('Feature importance (Average LDA class means across CV folds)');
xlabel('Voxel index'); 
ylabel('|Mean difference|');
legend('Average LDA feature difference');
