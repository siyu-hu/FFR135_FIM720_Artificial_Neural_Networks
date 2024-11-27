%FFR135 HM2.1 Perceptron with one hidden layer
%Author: Siyu Hu 19950910-3702 gushusii@student.gu.se

clear all;
close all;

% Preprocessing dataset
[x_train, t_train, x_val, t_val] = preprocess_data('training_set.csv', 'validation_set.csv');

% Initialize network

M1 = 10; % # of hidden neurons
input_dim = 2;
learning_rate = 0.002; 
epochs = 1000;   
batch_size = 100;

w1 = randn(M1, input_dim) * (1/sqrt(input_dim));  % weights of input-> hidden 
theta1 = zeros(M1, 1);     % threshold of hidden  
w2 = randn(1, M1) * (1/sqrt(M1));     % weights of hidden -> output     
theta2 = 0;     % threshold of output

num_samples = size(x_train, 1);
energy_values = zeros(epochs, ceil(num_samples/batch_size));


for epoch = 1:epochs

    for batch = 1:ceil(num_samples / batch_size)

        start_index = (batch - 1) * batch_size + 1;
        end_index = min(batch * batch_size, num_samples);

        x_batch = x_train(start_index:end_index, :);
        t_batch = t_train(start_index:end_index);

        % Training network
        hidden_layer_output = tanh(bsxfun(@minus, w1 * x_batch', theta1)); 
        output_layer_output = tanh(w2 * hidden_layer_output - theta2); 
    
        % Backpropagation
        output_error = (t_batch' - output_layer_output) .* (1 - output_layer_output.^2); 
        hidden_layer_error = (w2' * output_error) .* (1 - hidden_layer_output.^2);
    
        % Update Weights and threshold
        w2 = w2 + learning_rate * (hidden_layer_output * output_error')';
        theta2 = theta2 - learning_rate * sum(output_error); 
    
        w1 = w1 + learning_rate * (hidden_layer_error * x_batch);
        theta1 = theta1 - learning_rate * sum(hidden_layer_error, 2);
        
        % Compute Energy function
        energy_values(epoch, batch) = 0.5 * sum((t_batch' - output_layer_output).^2);
    
        % DEBUG
        %fprintf('Epoch %d, Batch %d: Energy = %f\n', epoch, batch, energy_values(epoch, batch));

    end
    
    % Use Validation Dataset to compute Classification erros - C 
    V_val = tanh(bsxfun(@minus, w1 * x_val', theta1)); % hidden layer output
    O_val = tanh(w2 * V_val - theta2);  % ouput layer output
    predictions = sign(O_val);                      
    C = sum(abs(predictions - t_val')) / (2 * length(t_val)); 

  
    if C < 0.12
        fprintf('The trainng stops after %d th epoch，Classification error C = %6f\n', epoch, C);
        break;
    end
    
    % DEBUG
    fprintf('Epoch %d, C = %6f\n',  epoch, C);
end

    if C > 0.12
        fprintf('All epochs training stops and C is still over 0.12 \n')
    end
    

figure;
plot(1:epochs, energy_values(1:epochs), 'LineWidth', 2);
xlabel('Epochs');
ylabel('Energy function H');
title('Energy value of network with each epoch');

csvwrite('w1.csv', w1);
csvwrite('w2.csv', w2);
csvwrite('t1.csv', theta1);
csvwrite('t2.csv', theta2);

% DEBUG
fprintf('Parameters saved \n');

