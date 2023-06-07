function [distance] = smm(theta)
    persistent moment
    persistent matrix
    if isempty(moment)
        moment = load("moment.mat").data;
        matrix = 1./power(moment,2);
        matrix = diag(matrix);
    end
    sim_size = 2000;
    t = 16;
    %data = randn(1000, 1) * 3 + 2;
    %sim_data = randn(sim_size, 1) * theta(2) + theta(1);
    %mean_data = mean(data);
    %std_data = std(data);
    %mean_sim = mean(sim_data);
    %std_sim = std(sim_data);
    delta_p = theta(1);
    sigma_1 = theta(2);
    sigma_2 = theta(3);

    z_p = normrnd(0, sigma_1, [1 sim_size * t]);
    z_q = normrnd(0, sigma_2, [1 sim_size * t]);
    sim_moment = zeros(1, length(moment));
    
    for i = 1: 15
        if i == 1
            simu = z_p(1+(i-1)*sim_size: i*sim_size);
            z_p(1+(i-1)*sim_size: i*sim_size) = z_p(1+(i-1)*sim_size: i*sim_size) * delta_p;
        else
            simu = z_p(1+(i-2)*sim_size: (i-1)*sim_size)*delta_p + z_p(1+(i-1)*sim_size: i*sim_size);
            z_p(1+(i-1)*sim_size: i*sim_size) = z_p(1+(i-2)*sim_size: (i-1)*sim_size)*delta_p + z_p(1+(i-1)*sim_size: i*sim_size);
        end
        simu = simu + z_q(1+(i-1)*sim_size: i*sim_size);
        sim_moment(2*i-1: 2*i) = [mean(simu)  (mean(power(simu, 2)) /  mean(simu))];
    end

    % compute distance
    distance = (moment - sim_moment) * ...
               matrix * ...
               transpose(moment - sim_moment);
end