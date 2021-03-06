function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Unpack the y matrix into a set of logical vectors
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);
k = size(y_matrix, 1);

%Add a column for bias units of our samples in X
a1 = [ones(m, 1) X];

%Calculate z2: the activations for our input layer, and the inputs for
%g(z^(2))
z2 = a1*Theta1';

%Compute the activations for our hidden layer
a2 = sigmoid(z2);

%Add a column for the bias unit in our hidden layer
a2 = [ones(m, 1) a2];

%Calculate our inputs for g(z^(3))  - our output layer
z3 = a2*Theta2';

%Compute the activations for our output layer
a3 = sigmoid(z3);

% Calculate cost: this is done by element wise multiplication of y by the 
% sigmoid function of our ouput activations, taking the double sum of the
% result and dividing by m
J_for_y_eq_1 = (-1.*y_matrix) .* log(a3);
J_for_y_eq_0 = (1-y_matrix) .* log(1-a3);
J = J_for_y_eq_1 - J_for_y_eq_0;
J = sum(J);
J = sum(J);
J = J./ m;

%Calculate regularisation term: the sum of each theta value squared 
%multiplied by lambda / (2m) . 
%Theta_0 for our bias units are excluded from this calculation
%
%The specification gives that there are exactly 3 layers.
regularisation_1 = Theta1(:, 2:end);
regularisation_2 = Theta2(:, 2:end);
regularisation_1 = regularisation_1.^2;
regularisation_2 = regularisation_2.^2;
regularisation_1 = sum(regularisation_1);
regularisation_1 = sum(regularisation_1);
regularisation_2 = sum(regularisation_2);
regularisation_2 = sum(regularisation_2);
regularisation = (regularisation_1 + regularisation_2) .* (lambda)/(2.*m);

%Apply regularisation term to our cost value
J = J + regularisation;

% Calculate error for output layer as the difference between output layer 
% activation matrix: a3 and training solution: y_matrix
d3 = a3-y_matrix;

% Calculate error for hidden layer as the output layer error: d3 
% multiplied by our weights for the hidden layer: Theta2 excluding the
% weights for the bias unit in the first column
% The result of the matrix multiplication above is then multiplied element-
% wise by the sigmoid gradient of the input for layer 3: z2
d2 = d3*(Theta2(:, 2:end));
d2 = d2 .* sigmoidGradient(z2);

% Calculate Delta for current layer by multiplying error: d of next layer: 
% d^(i+1) multiplied by activation: a of current layer: a^(i) working from
% final layer l back to initial layer 1.
Delta2 = d3'*a2;
Delta1 = d2'*a1;

% Calculate the gradients by taking each Delta and scaling it by 1/m
Theta2_grad = Delta2 ./ m;
Theta1_grad = Delta1 ./ m;

% Create regularisation terms: 
% Multiply each Theta by lambda/m. 
% Do not regularise bias units, so set column 1 to = 0 of our
% regularisation term
Theta2_regularisation = Theta2 .* (lambda/m);
Theta2_regularisation(:, 1) = 0;
Theta1_regularisation = Theta1 .* (lambda/m);
Theta1_regularisation(:, 1) = 0;

% Apply regularisation terms to gradients
Theta2_grad = Theta2_grad+Theta2_regularisation;
Theta1_grad = Theta1_grad+Theta1_regularisation;


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
