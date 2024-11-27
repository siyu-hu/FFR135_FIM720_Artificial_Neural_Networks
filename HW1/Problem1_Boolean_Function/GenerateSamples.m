% Generate random samples and avoid generating all combinations directly for large n
% because the number of unique samples is 2(2^n)）

function input_samples = GenerateSamples(n, num_samples)

    % Calculate the number of rows per sample (2^n combinations of inputs)
    rows_per_sample = 2^n; 

    % Preallocate cell array to store samples
    input_samples = cell(num_samples, 1);

    % Generate all input combinations (2^n possibilities)
    x = dec2bin(0:rows_per_sample - 1) - '0'; % Generate input matrix of size (2^n) x n

    % For each sample, randomly generate outputs and combine with inputs
    for i = 1:num_samples
        %CASE1 'random' % Completely random targets (-1 or 1), size (2^n) x 1
        target_outputs = randi([0, 1], rows_per_sample, 1) * 2 - 1; 

        %CASE2 'alternating' % Alternating -1 and 1
        %target_outputs = repmat([-1; 1], rows_per_sample/2, 1);

        % Combine inputs and targets into a single sample
        input_samples{i} = [x, target_outputs];
        
        % Check generation
        if isempty(input_samples{i})
            fprintf('Error: input_samples{%d} is empty!\n', i);
        end
             
    end
    
    % Final check of filled samples
    if isempty(input_samples)
        fprintf('Error: input_samples array is empty after generation!\n');
    else
        fprintf('Sample generation completed successfully.\n');
    end
end




