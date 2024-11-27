function kl_div = compute_kl_divergence(W, visible_bias, hidden_bias, num_generated_samples)
    full_data = [
        -1 -1 -1;
        -1 -1  1;
        -1  1 -1;
        -1  1  1;
         1 -1 -1;
         1 -1  1;
         1  1 -1;
         1  1  1;
    ];

    full_target_prob = [ 0.25, 0 , 0 , 0.25 , 0 , 0.25, 0.25, 0 ];

    num_visible = size(W, 1);   

    generated_samples = zeros(num_generated_samples, num_visible);
   
    % 生成一个随机的初始 v 向量 (1 x 3)，每个元素是 -1 或 1
    v = 2 * (rand(1, num_visible) > 0.5) - 1;

    % Gibbs 采样达到稳定后，开始采样 num_generated_samples 个样本
    for i = 1:num_generated_samples
        [h, ~] = sample_hidden(v', W, hidden_bias);
        v = sample_visible(h, W, visible_bias)';
        generated_samples(i, :) = v; 
    end

    % compute distribution prob. of model-generated samples
    num_patterns = size(full_data, 1);
    model_prob = zeros(1, num_patterns); 

    for i = 1:num_patterns
        current_sample = full_data(i, :);
        matching_samples = sum(all(generated_samples == current_sample, 2));
        model_prob(i) = matching_samples / num_generated_samples; 
    end

    % DEBUG
    disp('Model Probabilities:');
    disp(model_prob);
    disp('Target Probabilities:');
    disp(full_target_prob); 

    % compute KL distribution
    kl_div = 0;
    for i = 1:num_patterns
        if full_target_prob(i) > 0 
            kl_div = kl_div + full_target_prob(i) * log(full_target_prob(i) / (model_prob(i) + eps));  
        end
    end
end
