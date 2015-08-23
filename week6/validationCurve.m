function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================

for i = 1:length(lambda_vec),

    lambda = lambda_vec(i);
    [theta] = trainLinearReg(X,y,lambda);

    % compute train/val errors when training linear regression with lambda
    % Pass lambda = 0 for non-regularized training error
    [Jtrain, grad_train] = linearRegCostFunction(X,y,theta,0);
    [Jval, grad_val] = linearRegCostFunction(Xval,yval,theta,0);

    error_train(i) = Jtrain;
    error_val(i) = Jval;

end

% =========================================================================

end
