function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%

%Gaussian Kernel:
%K_{gaussian}\left( x^{\left( i\right) },x^{\left( j\right) }\right) =e^\left( -\sum ^{n}_{k=1}\dfrac{\left( x_{k}^{\left( i\right) }-x_{k}^{\left( j\right) }\right) ^{2}}{2\sigma ^{2}}\right)
sim = exp( sum( (x1-x2).^2 ) .* (-1/(2*(sigma).^2)) );




% =============================================================
    
end
