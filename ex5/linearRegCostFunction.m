function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%{
Cost is defined as:
$$J\theta =\dfrac {1}{2m}\left( \sum ^{m}_{i=1}\left( h_{\theta }\left( 
x^{\left( i\right) }\right) -y^{\left( i\right) }\right) ^{2} \right) 
+\dfrac {\lambda }{2m}\left( \sum ^{n}_{j=1}\theta ^{2}j\right)$$

Where the 2nd sum is only applied to the second and following elements of
theta
%}

%Calculate cost
J = (1/(2*m)).* sum( ( (X * theta) - y) .^2 );

%Calculate & apply regularization for cost
if lambda>0
    costReg = (lambda/(2*m)) .* sum(theta(2:end).^2);
    J = J + costReg;
end






% =========================================================================

grad = grad(:);

end
