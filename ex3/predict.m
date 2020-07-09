function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% Add column of 1s for our bias unit for each case to create the input layer
% a^(1)
X = [ones(m, 1) X];

% z^(2) = Theta^(1)*a^(1)
% a^(2) = g(z^(2)) plus a column of ones for our bias unit
hiddenLayerOne = sigmoid(X*Theta1');
hiddenLayerOne = [ ones(size(hiddenLayerOne, 1), 1), hiddenLayerOne ];


% z^(3) = Theta^(2)*a^(2)
% a^(3) = g(z^(3))
outputLayer = sigmoid(hiddenLayerOne*Theta2');

% Find the indices of the maximum values for the output layer. 
% Max run along dimension two to produce a column vector: p
[Y p] = max(outputLayer, [], 2);




% =========================================================================


end
