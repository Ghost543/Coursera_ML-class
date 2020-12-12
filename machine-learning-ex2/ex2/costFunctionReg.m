function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
h_x=sigmoid(X*theta);
% You need to return the following variables correctly 
J = 0;

grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
shift_theta = theta(2:size(theta));
 %X_0=X(:,1);
 %l=size(X)(1,2);

 %X_rest=X(:,2:l);
J=((1/m)*sum(-y'*log(h_x)-(1-y)'*log(1-h_x))+(lambda/(2*m))*sum([0;shift_theta].^2));

 %grad_0 = sum((h_x- y)'*X_0)/m;
%grad_rest=sum((h_x - y)'*X_rest)/m + (lambda/m)*(shift_theta);

%grad=(1/m)*sum((h_x-y)'*X) + (lambda/m)*[0;shift_theta];
grad = (1/m)*(X'*(h_x-y)+lambda*[0;shift_theta]);


% =============================================================

end
