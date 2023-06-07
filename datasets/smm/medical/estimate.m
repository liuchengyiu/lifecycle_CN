%theta0 = [0.904839, 0.943718, 0.079312];
theta0 = [0.904839, 0.3718, 0.79312];
options = optimoptions(@fmincon, 'display', 'iter', 'algorithm', 'interior-point');
[theta_hat, fval] = fmincon(@smm, theta0, [], [], [], [], [0 0.9 0 -1 -1 0 0], [2 1.1 1 1 1 1 1], [], options);

fprintf('The estimated parameters are: %f\n', theta_hat);
fprintf('The minimum distance is: %f\n', fval);
