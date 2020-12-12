function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters,
   theta_copy = theta';
   h_x = theta_copy.*X; 
   theta = [(theta_copy(:,1) - alpha*(sum(h_x(:,1) + h_x(:,2) - y))/(m)),; (theta_copy(:,2) - alpha*(sum((h_x(:,1) + h_x(:,2) - y).*(X(:,2))))/(m))];
   %theta = theta_copy - alpha*(sum((theta_copy'.*X-y).*X))/(m); 
   %theta = theta(:,1);
   %theta = (theta_copy' - alpha*(sum((theta_copy'.*X-y).*X))/(m))';
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
