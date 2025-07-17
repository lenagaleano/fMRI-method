%% Group-Level Inference Script: Paired t-test for Incongruent > Congruent

% Based on Verstynen et al. (2014), the following regions are linked to Stroop-related cognitive control:
% DLPFC, ACC, mOFC, Striatum

clear; clc;

%% Load MNI mask
mask_file = 'sub-002_task-stroop_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz';
mask_info = niftiinfo(mask_file);
mask = logical(niftiread(mask_file));
[nx, ny, nz] = size(mask);
n_subjects = 12;

%% Define ROI
roi_centers = [45,65,40; 46,60,45; 45,70,35; 48,55,35];
roi_mask = zeros(nx, ny, nz);
for i = 1:size(roi_centers,1)
    cx = roi_centers(i,1); cy = roi_centers(i,2); cz = roi_centers(i,3);
    roi_mask(cx-1:cx+1, cy-1:cy+1, cz-1:cz+1) = 1;
end

roi_smooth = imgaussfilt3(single(roi_mask), 2);
roi_template = imresize3(roi_smooth, [nx ny nz]);
roi_template = roi_template / max(roi_template(:));

%% Simulate activation maps for two conditions
%  Condition 1: Congruent, lower activation
%  Condition 2: Incongruent, higher activation
map_cong = NaN(nx, ny, nz, n_subjects);
map_incong = NaN(nx, ny, nz, n_subjects);

for s = 1:n_subjects
    noise1 = randn(nx, ny, nz)*0.3;
    noise2 = randn(nx, ny, nz)*0.3;
    scale_cong = 1.0 + 0.1*randn;
    scale_incong = 1.4 + 0.1*randn;
    map_cong(:,:,:,s) = scale_cong * roi_template + noise1;
    map_incong(:,:,:,s) = scale_incong * roi_template + noise2;
end

% Mask NaNs
map_cong(~mask) = NaN;
map_incong(~mask) = NaN;

%% Paired t-test: Incongruent > Congruent
% Test H0: mean(incongruent - congruent) = 0
% Test H1: mean(incongruent - congruent) > 0

diff_map = map_incong - map_cong;  % subject-wise differences
mean_diff = mean(diff_map, 4, 'omitnan');
std_diff = std(diff_map, 0, 4, 'omitnan');
t_map = mean_diff ./ (std_diff / sqrt(n_subjects));
t_map(~mask) = NaN;

%% Thresholds
df = n_subjects - 1;
alpha = 0.01;  % one-sided test
t_thresh = tinv(1 - alpha, df);

t_map_thr = t_map;
t_map_thr(t_map_thr < t_thresh) = NaN;

slice_num = round(nz / 2);

%% Plot: Unthresholded t-map
figure('Color','w');
imagesc(t_map(:,:,slice_num), [-4 6]);
axis image off; colormap(turbo); colorbar;
title('t-map: Incongruent > Congruent (Unthresholded)');

%% Plot: Thresholded t-map
figure('Color','w');
imagesc(t_map_thr(:,:,slice_num), [-4 6]);
axis image off; colormap(turbo); colorbar;
title(sprintf('Thresholded: t > %.2f (p<%.2f)', t_thresh, alpha));
