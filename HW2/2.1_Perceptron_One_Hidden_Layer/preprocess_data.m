function [x_train, t_train, x_val, t_val] = preprocess_data(training_file, validation_file)

    training_data = csvread(training_file);
    validation_data = csvread(validation_file);

    x_train = training_data(:, 1:2); 
    t_train = training_data(:, 3);   

    x_val = validation_data(:, 1:2);
    t_val = validation_data(:, 3);   

    % standarlize dataset 
    mean_train = mean(x_train);
    std_train = std(x_train);
    x_train = (x_train - mean_train) ./ std_train; 

    % using train mean and std (contain info of training data)processing validation, 
    % is aim to prevent data leasing
    x_val = (x_val - mean_train) ./ std_train;    
end
