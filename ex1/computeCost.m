function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta) %number of features

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%Compute element-wise summation of (theta * x - y)^2
%for i = 1:m
%    for j = 1:n
%        J = J + theta(j)*(X(i, j)-y(i))^2;
%    end
%end

%Vectorised method, 1 seperate line per operation
J = X*theta;
J = (J-y);
J = J.^2
J = J/(2*m);
J = sum(J)



% =========================================================================

end
