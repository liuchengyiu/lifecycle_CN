function [samples] = gmm(mu1, mu2, sigma1, sigma2, p1, num)
    sigma = cat(3, [sigma1], [sigma2]);
    mu = [mu1; mu2];
    p = [p1 1-p1];
    gm = gmdistribution(mu,sigma,p);
    samples = random(gm,num);
end