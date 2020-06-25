function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

%{
We can expect 
    X: to be a (m x n+1) matrix of m examples and n+1 features
    y: to be a (m x 1) column vector for m examples
    theta: to be a (n+1 x 1) row vector for n+1 features
    alpha: to be a positive real number
    num_iters: to be a positive integer
    
    some handy reference material: 
    https://www.coursera.org/learn/machine-learning/supplement/U90DX/gradient-descent-for-linear-regression
    https://www.coursera.org/learn/machine-learning/supplement/WKgbA/multiple-features
%}

% Initialize some useful values
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    clc
    debug = false;

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    


    if debug
        alpha
        theta
        X
        X*theta
        y
        
    end
    
    %Assign variables for the dimensionality of observations & features in
    %the data
    [m n] = size(X);
    
    
    %{
    Method with loops
   
    
    
    %Initialise the updates to be 0
    %update_1_with_loops = 0
    %update_2_with_loops = 0
        
        
    %Find the derivative 
    for i = 1:m
        for j = 1:n
            update_1_with_loops = update_1_with_loops + ( (X(j) .* theta(j)) - y(j) ) .* X()
        end
    end
     %}
    
    %End method with loops, rather than vectorisation
    
    %Method with vectorisation, rather than loops
    update = ((X*theta)-y)'*X;
    
    update = update.*(alpha./m);
    
    if debug
        print(update)
    end
    
    theta = theta - update';


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
