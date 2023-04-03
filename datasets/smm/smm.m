% SMM
function [distance] = smm(theta)
    persistent moment
    persistent matrix
    if isempty(moment)
        moment = load("moment.mat").data;
        matrix = 1./power(moment,2);
        matrix = diag(matrix);
    end
    sim_size = 2000;
    t = 41;
    %data = randn(1000, 1) * 3 + 2;
    %sim_data = randn(sim_size, 1) * theta(2) + theta(1);
    %mean_data = mean(data);
    %std_data = std(data);
    %mean_sim = mean(sim_data);
    %std_sim = std(sim_data);
    sigma_q = theta(1);
    delta_p = theta(2);
    p1 = theta(3);
    mu_1 = theta(4);
    mu_2 = theta(5);
    sigma_1 = theta(6);
    sigma_2 = theta(7);

    z_q = normrnd(0, sigma_q, [1 sim_size * t]);
    eta = transpose(gmm(mu_1, mu_2, sigma_1, sigma_2, p1, sim_size*t));
    sim_moment = zeros(1, length(moment));
    
    for i = 1: 40
        if i == 1
            simu = eta(1+(i-1)*sim_size: i*sim_size);
            eta(1+(i-1)*sim_size: i*sim_size) = eta(1+(i-1)*sim_size: i*sim_size) * delta_p;
        else
            simu = eta(1+(i-2)*sim_size: (i-1)*sim_size)*delta_p + eta(1+(i-1)*sim_size: i*sim_size);
            eta(1+(i-1)*sim_size: i*sim_size) = eta(1+(i-2)*sim_size: (i-1)*sim_size)*delta_p + eta(1+(i-1)*sim_size: i*sim_size);
        end
        simu = simu + z_q(1+(i-1)*sim_size: i*sim_size);
        sim_moment(2*i-1: 2*i) = [mean(simu)  (mean(power(simu, 2)) /  mean(simu))];
    end

    % compute distance
    distance = (moment - sim_moment) * ...
               matrix * ...
               transpose(moment - sim_moment);
end