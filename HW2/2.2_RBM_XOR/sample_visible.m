% sample_visible.m
function visible_sample = sample_visible(hidden_sample, weights, visible_bias)

    visible_activation = weights * hidden_sample - visible_bias;  
    activation_prob_v = (tanh(visible_activation) + 1) / 2;
    visible_sample = 2 * (rand(size(visible_activation)) < activation_prob_v) - 1;  
end
