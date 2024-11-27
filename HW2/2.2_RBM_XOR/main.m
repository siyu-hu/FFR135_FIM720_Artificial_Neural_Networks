% FFR135 HM2.2 RBM with XOR dataset
% Author: Siyu Hu, 19950910-3702, gushusii@student.gu.se

% main.m

% Initialize data and target probabilities
data = [[-1 -1 -1]; [-1 1 1];[1 -1 1]; [1 1 -1]] ;
target_prob = [0.25, 0.25, 0.25 , 0.25];  

% Set Parameters
learning_rate = 0.001; 
k = 10;  
epochs = 1000;  
batch_size = 100;  
num_generated_samples = 100000;  
hidden_units = [1, 2, 4, 8];  
num_visible = size(data, 2); 
runs = 20;

kl_divergences = zeros(1, length(hidden_units));


%prepare to plot
figure;
hold on;
colors = lines(runs); 
legend_labels = cell(1, runs);

for n = 1: runs
for i = 1:length(hidden_units)
    M = hidden_units(i); 
    fprintf('Training with %d hidden units...\n', M);
    
    W = (1 / sqrt(max(3, M)))  * randn(num_visible, M );
    visible_bias = zeros(size(data, 2), 1);  
    hidden_bias = zeros(M, 1);  

    % Training Network
    for epoch = 1:epochs
        delta_W = zeros(num_visible, M);
        delta_visible_bias = zeros(num_visible, 1);
        delta_hidden_bias = zeros(M, 1);

        for batch_idx = 1:batch_size
            % randomly choose a sample
            v0 = data(randi([1 size(data, 1)]), :)';  
            [h_initial, b_h_initial] = sample_hidden(v0, W, hidden_bias);

            % CD-k algorithm
            for step = 1:k
                v_k = sample_visible(h_initial, W, visible_bias);
                [h_k, b_h] = sample_hidden(v_k, W, hidden_bias);
            end
            delta_W = delta_W + learning_rate * (v0 * tanh(b_h_initial)' - v_k * tanh(b_h)');
            delta_visible_bias = delta_visible_bias - learning_rate * (v0 - v_k);
            delta_hidden_bias = delta_hidden_bias - learning_rate * (tanh(b_h_initial) - tanh(b_h));
        end

        W = W + delta_W ;
        visible_bias = visible_bias + delta_visible_bias ;
        hidden_bias = hidden_bias + delta_hidden_bias;
    end


    kl_div = compute_kl_divergence(W, visible_bias, hidden_bias, num_generated_samples);
    kl_divergences(i) = kl_div;  
    fprintf('KL divergence for %d hidden units: %.4f\n', M, kl_div);
end


plot(hidden_units, kl_divergences, 'o', 'LineWidth', 2, 'Color', colors(n, :));
legend_labels{n} = sprintf('Run %d', n);  

end 


% Plot theoretical KL_div boundary [Equation (4.40)]
N = 3;  
D_kl_theory = zeros(1, length(hidden_units));

for i = 1:length(hidden_units)
    M = hidden_units(i);
    if M < 2^(N-1) - 1
        term2 = (M + 1) / (2^(log2(M+1)));
        D_kl_theory(i) = log(2) * (N - log2(M + 1) - term2);
    else
        D_kl_theory(i) = 0;  
    end
end

plot(hidden_units, D_kl_theory, '--r', 'LineWidth', 2); 
legend_labels{runs + 1} = 'Theoretical KL Divergence (Equation 4.40)';  

xlabel('Number of Hidden Units (M)');
ylabel('KL Divergence (D_{KL})');
title('KL Divergence vs. Number of Hidden Units');
grid on;

legend(legend_labels, 'Location', 'best');

hold off;