function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

%Compute values for the negative and the positive case of y
J_for_y_eq_1 = (-1.*y) .* ( log(sigmoid (X * theta) ) );
J_for_y_eq_0 = (1-y) .* log( 1 - (sigmoid (X * theta) ) );

%Subtract the results of the negative cases from the positive cases to
%produce our cost vector for each case
J = J_for_y_eq_1 - J_for_y_eq_0;

%Create a constant for regularisation, taking lambda/m and theta as
%the arguments
c_regular = sum(theta.^2);
c_regular = c_regular .* (lambda/(2.*m)); 

%Apply regularisation vector to our cost vector
%BUT! Only to the 2nd and subsequent elements
J(2:end) = J(2:end)+c_regular;

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

grad = grad(:);

end
