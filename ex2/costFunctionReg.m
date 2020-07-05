function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
%m = number of training examples
%n = number of features
[m, n] = size(X);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


%Compute values for the negative and the positive case of y
J_for_y_eq_1 = (-1.*y) .* ( log(sigmoid (X * theta) ) );
J_for_y_eq_0 = (1-y) .* log( 1 - (sigmoid (X * theta) ) );

%Subtract the results of the negative cases from the positive cases to
%produce our cost vector for each case
J = J_for_y_eq_1 - J_for_y_eq_0;

%Create a constant for regularisation, taking lambda/m and theta as
%the arguments
cost_regularisation_constant = sum(theta(2:end, :).^2) .* (lambda/(2.*m));
%{
A more spread out version of the above line:
cost_regularisation_constant = theta(2:end, :);
cost_regularisation_constant = cost_regularisation_constant.^2;
cost_regularisation_constant = sum(cost_regularisation_constant);
cost_regularisation_constant = cost_regularisation_constant .* (lambda/(2.*m)); 
%}


%Apply regularisation vector to our cost vector
%BUT! Only to the 2nd and subsequent elements
J(2:end) = J(2:end)+cost_regularisation_constant;

% sum our cost vector J,then divide by number of cases to get the mean value
J = sum(J) ./ m;

%Calculate the gradient
grad = ( (sigmoid(X*theta)-y)'*X ./m );


%Build regularisation vector for gradient: \frac{\lambda}{m}\theta_j, 
%applies to elements 2-m, so not to apply the regularisation to the bias 
%element
grad_regularisation = theta';
grad_regularisation(1) = 0;
grad_regularisation = grad_regularisation .* (lambda./m);


%Add regularisation vector to gradient vector
grad = grad + grad_regularisation;

% =============================================================

end
