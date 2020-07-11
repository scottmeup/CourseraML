function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms
%

% You need to return the following variables correctly 
%W = zeros(L_out, 1 + L_in);

% ====================== YOUR CODE HERE ======================
% Instructions: Initialize W randomly so that we break the symmetry while
%               training the neural network.
%
% Note: The first column of W corresponds to the parameters for the bias unit
%

%suggested weight for random values is 0.12
epsilon = 0.12;

%Initialise elements to a random value between 0 and 1
W = rand(L_out, 1 + L_in);

%Manipilate the elements of W so that -epsilon <= W_ij <= +epsilon for all
%W_ij
W = W.*2.*epsilon;
W = W-epsilon;


% =========================================================================

end
