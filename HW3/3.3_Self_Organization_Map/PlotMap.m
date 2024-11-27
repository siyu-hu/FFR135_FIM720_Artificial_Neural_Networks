function PlotMap(weights, data, labels)
    hold on;
    mapSize = size(weights, 1);  
    nFeatures = size(weights, 3);  

    pointSize = 100; 
    for i = 1:size(data, 1)
        inputVector = data(i, :);
        reshapedInput = reshape(inputVector, [1 1 nFeatures]);
        difference = weights - reshapedInput;
        distances = sum(difference.^2, 3);
        [~, winnerIdx] = min(distances(:));
        [winnerRow, winnerCol] = ind2sub([mapSize, mapSize], winnerIdx);

       
        if labels(i) == 1
            scatter(winnerRow, winnerCol, pointSize, 'r', 'filled');  
        elseif labels(i) == 2
            scatter(winnerRow, winnerCol, pointSize, 'g', 'filled');  
        else
            scatter(winnerRow, winnerCol, pointSize, 'b', 'filled');  
        end
    end

    scatter(NaN, NaN, pointSize, 'r', 'filled');  
    scatter(NaN, NaN, pointSize, 'g', 'filled');  
    scatter(NaN, NaN, pointSize, 'b', 'filled'); 

    hold off;
end