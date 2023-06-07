% init_p = [0.0692186255344865 1.0002244596913306 0.020148176217138148 0.0011265992581365625 0.44902496866025565 0.19721901813879006 0.2166529381789416];
% zq_std persistent mu1 v1 v2 p1 p2
% theta0 = zeros([1 7]); % init
% theta0(1) = init_p(1);
% theta0(2) = init_p(2);
% theta0(3) = init_p(end-1);
% theta0(4) = init_p(3) * init_p(end);
% theta0(5) = init_p(3) * init_p(end-1);
% theta0(6) = init_p(4);
% theta0(7) = init_p(5);

theta0 = [0.6 0.998, 0.3, 0.2, -0.2, 1, 1];

options = optimoptions(@fmincon, 'display', 'iter', 'algorithm', 'interior-point');
[theta_hat, fval] = fmincon(@smm, theta0, [], [], [], [], [0 0.9 0 -1 -1 0 0], [2 1.1 1 1 1 1 1], [], options);

fprintf('The estimated parameters are: %f\n', theta_hat);
fprintf('The minimum distance is: %f\n', fval);
% distance 179 param [0.477935 0.900000 0.221104 0.027078 -0.138412 0.999957 0.022332]
% distance 157 param [0.793809 0.925428 0.232455 -0.687735 0.120065 0.714885 0.002731]