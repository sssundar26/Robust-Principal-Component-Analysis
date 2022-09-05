clear all;
clc;
load('Image_anomaly.mat')

figure;
imagesc(X);

[m n] = size(X); 
tol = 1e-7; 
lambda=1e-2;

% Initialization
Y = X; 
norm_two = norm(Y); 
norm_inf = norm( Y(:), inf) / lambda;
Y = Y / norm_inf;
A_hat = zeros( m, n); 
E_hat = zeros( m, n);
mu = 0.5/norm_two;
mu_bar = mu * 1e7; 
rho = 1.3;         % Parameter for tuning
d_norm = norm(X, 'fro');
iter = 0; 
total_svd = 0; 
converged = false;
stopCriterion = 1;
%ADMM updates
while ~converged
    iter = iter + 1;
    temp_T = X - A_hat + (1/mu)*Y;
    E_hat = max(temp_T - lambda/mu, 0);
    E_hat = E_hat+min(temp_T + lambda/mu, 0);
    [U S V] = svd(X - E_hat + (1/mu)*Y, 'econ');
    diagS = diag(S);
    svp = length(find(diagS > 1/mu));
    A_hat = U(:, 1:svp) * diag(diagS(1:svp) - 1/mu) * V(:, 1:svp)';
    total_svd = total_svd + 1;
    Z = X - A_hat - E_hat;
    Y = Y + mu*Z;
    mu = min(mu*rho, mu_bar);
    % stop Criterion
    stopCriterion = norm(Z, 'fro') / d_norm;
    if stopCriterion < tol
        converged = true;
    end
end

figure; imagesc((A_hat))
figure;imagesc(E_hat)