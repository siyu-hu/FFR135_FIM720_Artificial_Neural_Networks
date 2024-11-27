% FFR105 HW3.3 Self organising map
% Author: Siyu Hu, gushusii@student.gu.se

function TrainSOM()
    clear all;
    clc;
    
    data = csvread('iris_data.csv');
    labels = csvread('iris_labels.csv');
    data = data ./ max(data(:)); 

    % Initialize Weights
    mapSize = 40;
    nFeatures = 4;
    weights = rand(mapSize, mapSize, nFeatures);
    initialWeights = weights;

    eta0 = 0.1;
    dEta = 0.01;
    sigma0 = 10;
    dSigma = 0.05;
    epochs = 10;

    figure;
    subplot(1, 2, 1);  
    title('Initial Weights');
    PlotMap(initialWeights, data, labels);

    for epoch = 1:epochs
        % Update learning rate and width of neighbourhood
        fprintf("Epoch = %d \n", epoch);
        eta = eta0 * exp(-dEta * epoch);
        sigma = sigma0 * exp(-dSigma * epoch);

        for i = 1:size(data, 1)
            inputVector = data(i, :);

            % find winner neuron
            reshapedInput = reshape(inputVector, [1 1 nFeatures]);
            difference = weights - reshapedInput;
            distances = sum(difference.^2, 3);
            [~, winnerIdx] = min(distances(:));
            [winnerRow, winnerCol] = ind2sub([mapSize, mapSize], winnerIdx);

            % update weights
            for r = 1:mapSize
                for c = 1:mapSize
                    distSq = (r - winnerRow)^2 + (c - winnerCol)^2;
                    h = exp(-distSq / (2 * sigma^2));  % neighbourhood function
                    currentWeight = squeeze(weights(r, c, :))';
                    deltaWeight = eta * h * (inputVector - currentWeight);
                    newWeight = currentWeight + deltaWeight;
                    weights(r, c, :) = reshape(newWeight, [1, 1, nFeatures]);
                end
            end
        end
    end

    % Plot
    subplot(1, 2, 2);  
    title('Final Weights');
    PlotMap(weights, data, labels);
end
