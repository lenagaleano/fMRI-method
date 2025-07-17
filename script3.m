%% 3. Stroop GLM – Exercises 
clear; clc;

%% 1. Load reference brain mask info
mask_file = 'sub-002_task-stroop_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz';
mask_info = niftiinfo(mask_file);
mask_data = logical(niftiread(mask_file));  % binary mask

%% 2. Define scan parameters
[nx, ny, nz] = size(mask_data);
nt = 160;
TR = 2;
timepoints = (0:nt-1)' * TR;
slice_num = round(nz / 2);

%% 3. Simulate fMRI data and apply brain mask
Y = randn(nx, ny, nz, nt);  % synthetic 4D data
for t = 1:nt
    volume = Y(:,:,:,t);
    volume(~mask_data) = NaN;  % set non-brain voxels to NaN
    Y(:,:,:,t) = volume;
end

%% 4. Load actual Stroop events
T = readtable('stroop_events.tsv', 'FileType', 'text');
idx_cong = round(T.onset(strcmp(T.trial_type, 'congruent')) / TR) + 1;
idx_incong = round(T.onset(strcmp(T.trial_type, 'incongruent')) / TR) + 1;

stim_cong = zeros(nt, 1); stim_incong = zeros(nt, 1);
stim_cong(idx_cong(idx_cong <= nt)) = 1;
stim_incong(idx_incong(idx_incong <= nt)) = 1;

%% 5. Create HRF and convolve
t_hrf = 0:TR:32;
hrf = gampdf(t_hrf,6,1) - 0.5*gampdf(t_hrf,12,1);
hrf = hrf / max(hrf);

reg_cong = conv(stim_cong, hrf); reg_cong = reg_cong(1:nt);
reg_incong = conv(stim_incong, hrf); reg_incong = reg_incong(1:nt);
X = [reg_cong, reg_incong, ones(nt,1)];

%% E1: Plot the design matrix
figure;
plot(timepoints, X(:,1:2)); xlabel('Time (s)');
legend({'Congruent','Incongruent'});
title('E1: Design Matrix (Regressors)');
grid on;

%% E2: Compute contrast Congruent > Incongruent
Y_2D = reshape(Y, [], nt)';  % time x voxel
betas = (X' * X) \ (X' * Y_2D);

c = [1 -1 0]';  % contrast vector
c_beta = c' * betas;
res = Y_2D - X * betas;
df = nt - rank(X);
mse = sum(res.^2,1) / df;
c_var = c' * inv(X'*X) * c;
t_vals = c_beta ./ sqrt(mse * c_var);
t_map = reshape(t_vals, nx, ny, nz);

%% E3: Show t-map slice for Congruent > Incongruent
figure;
imagesc(t_map(:,:,slice_num)); axis image; colorbar;
title('E3: t-map (Congruent > Incongruent)');


%% E4. Apply contrast: Incongruent > Congruent
%...