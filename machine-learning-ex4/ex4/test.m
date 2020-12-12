%mapping ys to there resspective Xs
I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
  Y(i, :)= I(y(i), :);
end

a=[ones(m,1) X];
z2=a*Theta1';
a2=sigmoid(z2);
a2=[ones(m,1) a2];
z3=a2*Theta2';
h_x = sigmoid(z3);

J = sum(sum(-Y.*log(h_x) + (1-y).*log(1-h_x)))*(1/m);

regu = (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:,2:end).^2, 2)));
J = J + regu;

for t=1:m,
  a_1=[1;X(t,:)'];
  z_2=Theta1*a_1;
  a_2=[1;sigmoid(z_2)];
  z_3=Theta2*a_2;
  a_3=sigmoid(z_3);
  
  
  delta_3=a_3-Y(t,:)';
  delta_2 = (Theta2'*delta_3).*[1;sigmoidGradient(z_2)];
  delta_2 = delta_2(2:end);
  %disp((delta_2(1,:)));

  Theta1_grad= Theta1_grad + delta_2*a_1';
  Theta2_grad= Theta2_grad + delta_3*a_2';
  
endfor

Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;
