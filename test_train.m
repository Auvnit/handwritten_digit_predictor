function [Theta1,Theta2] = test_train()
load('ex4data1.mat');
m = size(X, 1)
load('ex4weights.mat');
sel = randperm(size(X, 1));
sel1 = sel(1:3500);
sel2= sel(3500:end);
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
nn_params = [Theta1(:) ; Theta2(:)];
lambda = 3;
checkNNGradients(lambda);
options = optimset('MaxIter', 50);
lambda = 1;
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X(sel1,:), y(sel1,:), lambda);
initial_nn_params=nn_params;
[nn_params, ~] = fmincg(costFunction, initial_nn_params, options);
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
pred = predict(Theta1, Theta2, X(sel2,:));
rem(pred(:),10);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == y(sel2,:))) * 100);
end