% Main Function 
num_samples = 10000;    % Number of Boolean functions
num_epochs = 20;        % Number of training epochs
learning_rate = 0.05;   % Learning rate for traing perceptron
input_samples = cell(num_samples,1);
proportion = [0,0,0,0];


% Iterate over each dimension n
for n = 2:5

    % Generate samples for the current dimension
    input_samples = GenerateSamples(n, num_samples);
    count_linearly_separable = 0;
    unique_functions = {};  % Initialize cell array
    threshold = 0;
    weights = rand(1, n) *  sqrt(1 / n); 
    outputs = zeros(2^n, 1);
    
    
    for i = 1: num_samples
        % Select one sample 
        sample = cell2mat(input_samples(i));   
        inputs = sample(:,1:end-1);% the first n colunm is x1...xn
        target_outputs = sample(:, end); % the last column is  t

        % Count Unique target_ouputs 
        target_string = mat2str(target_outputs'); % preparing for comparing with member of unique_functions 
        % Check if this function has already been counted
        if ~ismember(target_string, unique_functions)
            unique_functions{end + 1} = target_string;% Add the target string to the end
        else
             continue;
        end
                 
        % Caculate initial outputs size = (2^n, 1)
        % AND Update weight matrix and threshold
        for k = 1: num_epochs
            for j = 1: size(inputs, 1)
                tempX= sign(weights * inputs(j,:)' - threshold);
                if tempX == 0
                   tempX = -1;
                end
                outputs(j) = tempX; 
            end
              
            for m = 1: length(target_outputs)    
                delta_weights = learning_rate * (target_outputs(m) - outputs(m)) * inputs(m , :);
                delta_threshold = -learning_rate * (target_outputs(m) - outputs(m));
                weights = weights + delta_weights;
                threshold = threshold + delta_threshold;
            end

            
        end
            
        % Check if the sample is linearly separable == the perceptron can solve the error in inputs == outputs is equal to targets_outputs
       
       outputs = sign(weights * inputs' - threshold * ones(1, 2^n));
       outputs = outputs';
 
       if outputs == target_outputs
         count_linearly_separable = count_linearly_separable + 1;
       end
    end

    if isempty(unique_functions)
       proportion(n-1) = 0;  % no unique functions are detected
    else
       proportion(n-1) = count_linearly_separable / length(unique_functions);
    end
end
   

    
  
disp('For n = 2,3,4,5, the proportion of linearly separable functions:');
disp(proportion);
%fprintf('Dimension n = %d, Proportion of linearly separable functions: %.4f\n', n, proportion);