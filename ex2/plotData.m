function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.


% Y = 1 (pass) or 0 (fail)
% X(:, 1) = Double (Test 1 Score)
% X(:, 2) = Double (Test 2 Score)

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

%My code below doesn't work too well with legends
%{

[m, ~] = size(X);

for i=1:m
    if y(i)==0
        %plot fail cases
        p1 = plot(X(i, 1), X(i, 2), 'ko', 'MarkerFaceColor', 'yellow', 'MarkerSize', 7);
        legend('Not Admitted')
    elseif y(i)==1
        %plot pass cases
        p2 = plot(X(i, 1), X(i, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
    end
    xlabel('Exam 1 Score')
    ylabel('Exam 2 Score')
end
legend([p2, p1], 'Admitted', 'Not Admitted')

%}

%Supplied code below
% Find Indices of Positive and Negative Examples
pos = find(y==1); neg = find(y == 0);
% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, ...
     'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', ...
     'MarkerSize', 7);



% =========================================================================



hold off;

end
