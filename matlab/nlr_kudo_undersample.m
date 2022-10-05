
clear all
close all
clc

path = 'C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\Kudo_NLR_image_data\NLR_image_data_sd_1.nii';
% savepath = 'C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\nlr_results\nlr_sd_0_subsampled05';

im = niftiread(path);
im = squeeze(im);
im = permute(im, [2 3 1 4]);
%%


%%

clearvars -except im
close all
clc
dt = 4;
baseline_until_ex = 3;
indices = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29];
aifcurves.unfiltered_rescaled = [40.863964, 39.63787, 39.831463, 41.154358, 496.7779, 293.0202, 101.23303, 49.86609, 42.47725, 39.476543, 40.928497, 40.121857, 41.38022, 39.18615, 41.638344]; %subsmapled, AIF geschaald, geen filter
%%
I0 = im(:, :, end-4:end, indices);
sz = size(I0);
%%
% Filter I0
% Ifilt = NaN(sz);
% for k = 1:sz(3)
%    for t = 1:sz(4)
%        Ifilt(:, :, k, t) = imgaussfilt(I0(:, :, k, t), 2.5);
%    end
% end
disp(sz(1))
disp(sz(2))
disp(sz(3))
% Run analysis
I = I0;
X = NaN(sz(1), sz(2), sz(3), 3);
aif = aifcurves.unfiltered_rescaled;
aif = aif'; aif = aif - mean(aif(1:baseline_until_ex));
%%
for i = 1:sz(1)
    disp(i/sz(1))
    for j = 1:sz(2)
        for k = 1:sz(3)
            
            tac = squeeze(I(i, j, k, :));
            if tac(1) < 10 || tac(1) > 50
                continue
            else
                tac = tac - mean(tac(1:baseline_until_ex));
            end
                        
            x = boxnlr(aif, tac, dt);
            X(i, j, k, :) = x;
            
        end
    end
end



% tac = squeeze(I(200, 200, 1, :)); tac = tac - mean(tac(1:5));

% figure()
% imshow3d_(squeeze(I), [0 80])

% figure()
% hold on
% box on
% plot(aif)
% plot(tac)

% x = boxnlr(aif, tac, dt);

%%
clearvars -except im X
close all
clc

PERF = X;

CBV = X(:, :, :, 1);
MTT = X(:, :, :, 2);
DEL = X(:, :, :, 3);

% figure()
% imshow(DEL, [0 5])
% colormap('jet')

PERF = squeeze(PERF);
niftiwrite(PERF, 'C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\nlr_results\nlr_sd_1_subsampled05')
% save(PERF, 'L:\basic\divi\CMAJOIE\CLEOPATRA\Substudies\Lucas\KudoPhantom\unfiltered_rescaled_aif_lucas.mat')

%%

% Estimate the perfusion parameters (x;
% CBV, MTT, and delay) that explain the
% tissue signal (tac) resulting from the
% input signal (aif). dt is the sample
% interval.
function x = boxnlr(aif, tac, dt)
% Initial estimates for CBV, MTT, and
% delay, respectively. Divide MTT and
% delay by dt seconds to convert to
% unitless values.
x0 = [0.05; 4/dt; 1/dt];
% Convolve AIF and TAC with a 3-point
% bandlimiting kernel with a FWHM of
% 2 samples. Handle edges by nearest
% neighbor extrapolation.
k = [0.25; 0.5; 0.25]; 
aif_k = conv([aif(1);aif;aif(end)], k, 'valid');
tac_k = conv([tac(1);tac;tac(end)], k, 'valid');
% Calculate the numerical integrand of
% the bandlimited AIF. Note that this
% cumulative sum introduces a half
% sample shift.
auc = cumsum(aif_k);
% Find optimal values. Fminsearch
% requires the MATLAB Optimization
% Toolbox. By default the method uses a
% Nelder-Mead simplex algorithm with a
% step size of 5% of the initial value.
f = @(x) fun(x, auc, tac_k);
x = fminsearch(f, x0);
% Multiply MTT and delay by dt seconds
% to convert from unitless values.
x = [x(1); x(2)*dt; x(3)*dt];
end
% Calculate the sum of squared errors
% (sse) between the measured TAC and the
% TAC generated from the AIF using the
% perfusion parameters in x.
function sse = fun(x, auc, tac_k)
tac_est = gen_tac(x(1)/x(2), x(2), x(3), auc);
sse = sum((tac_est - tac_k).^2);
end
% Generate a TAC from the AIF and a set
% of perfusion parameters. When a box-
% shaped impulse response is assumed,
% then the TAC can be calculated as the
% difference of shifted integrands of
% AIF. Note that an additional half
% sample shift is required to correct
% for the shift introduced by the
% cumulative sum. In case negative
% shifts should be handled correctly,
% then interp1 needs to extrapolate the
% right side of the integrated AIF.
function tac = gen_tac(cbf, mtt, d, auc)
n = length(auc);
a = interp1(auc, (1:n)'-0.5-d, 'linear', 0);
b = interp1(auc, (1:n)'-0.5-d-mtt, 'linear', 0);
tac = cbf * (a - b);
end