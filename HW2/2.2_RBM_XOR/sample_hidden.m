% sample_hidden.m
function [hidden_samle, hidden_activation] = sample_hidden(visible_sample, weights,  hidden_bias)
    hidden_activation = weights' * visible_sample - hidden_bias; 
    activation_prob_h = (tanh(hidden_activation) + 1) / 2;
    hidden_samle = 2 * (rand(size(hidden_activation)) < activation_prob_h) - 1;  
end
