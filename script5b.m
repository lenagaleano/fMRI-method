%% 2. Representational Similarity Analysis (RSA)
clear; clc; close all;
rng(25);

%% Parameters
nConditions = 4;   % (red-circle, green-circle, red-square, green-square)
nTrialsPerCond = 50; % total trials: 4 conditions * 50 trials each = 200 trials
nVoxels = 100;
nColorVoxels = 20;
nShapeVoxels = 20;

%% Labels for conditions
colors = repelem([1;2;1;2], nTrialsPerCond); % [R,G,R,G]
shapes = repelem([1;1;2;2], nTrialsPerCond); % [C,C,S,S]

%% Simulate voxel patterns
X = randn(nConditions * nTrialsPerCond, nVoxels);

% Add differences (color)
X(:, 1:nColorVoxels) = X(:, 1:nColorVoxels) + (colors - 1.5);

% Add differences (shape)
X(:, 21:20+nShapeVoxels) = X(:, 21:20+nShapeVoxels) + (shapes - 1.5);

%% Average patterns per condition with multiple trials
meanPatterns = zeros(nConditions, nVoxels);
for c = 1:nConditions
    meanPatterns(c,:) = mean(X((c-1)*nTrialsPerCond+1 : c*nTrialsPerCond, :), 1);
end

%% Compute neural RDM (1 - correlation)
neuralRDM = 1 - corr(meanPatterns');

%% Model RDMs
colorVec = [1;2;1;2];
shapeVec = [1;1;2;2];
model_color = double(colorVec ~= colorVec');
model_shape = double(shapeVec ~= shapeVec');

%% RSA correlations (excluding diagonals)
idx = logical(tril(ones(size(neuralRDM)), -1));
r_color = corr(neuralRDM(idx), model_color(idx));
r_shape = corr(neuralRDM(idx), model_shape(idx));
fprintf('RSA correlations: color = %.3f, shape = %.3f\n', r_color, r_shape);

%% Plot RDMs
figure;
subplot(1,3,1); imagesc(neuralRDM); title('Neural RDM'); axis square; colorbar;
subplot(1,3,2); imagesc(model_color); title('Color Model'); axis square; colorbar;
subplot(1,3,3); imagesc(model_shape); title('Shape Model'); axis square; colorbar;

