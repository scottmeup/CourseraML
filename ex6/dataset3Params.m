function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%list of values to use for C and sigma when trying to find the lowest error
valueList = [0.01 0.03 0.1 0.3 1 3 10 30];

%initialise variables to hold the value of our lowesr cost, the C, and 
%Sigma used to achieve that cost
minCost = inf;
C = 1;
sigma = 1;

%iterate over valueList to generate combinations of C and Sigma for
%modelling & cost comparison
[~, valueListDimension] = size(valueList);
for thisC = 1:valueListDimension
    for thisSigma = 1:valueListDimension
        %generate cost for this particular combination of C and Sigma
        thisModel = svmTrain(X, y, valueList(thisC), @(x1, x2) gaussianKernel(x1, x2, valueList(thisSigma)));
        thisPrediction = svmPredict(thisModel, Xval);
        thisCost = mean(double(thisPrediction ~= yval));
        
        %compare the cost of this iteration to lowest cost value previously
        %calculated.
        %store the values of thisCost, C, and Sigma if the cost for this 
        %iteration is lower than any previously generated cost.
        if thisCost < minCost
            minCost = thisCost;
            C = valueList(thisC);
            sigma = valueList(thisSigma);
        end
    end
end





% =========================================================================

end
