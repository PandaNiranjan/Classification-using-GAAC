function [rules, accuracy, FMeasure, AUC] = PART(trainData, testData)
    % PART (Partial Decision Trees) Algorithm
    % Arguments:
    % - trainData: Training dataset
    % - testData: Testing dataset
    % Returns:
    % - rules: Generated rules
    % - accuracy: Classification accuracy
    % - FMeasure: F-measure of the classifier
    % - AUC: Area Under the ROC Curve

    % Train the initial decision tree
    tree = fitctree(trainData{:, 1:end-1}, trainData{:, end});
    
    % Generate rules from the decision tree
    rules = generateRulesFromTree(tree);

    % Prune redundant rules
    rules = pruneRules(rules, trainData);

    % Test the rules on test data
    [predictions, scores] = applyRules(rules, testData);

    % Evaluate performance
    [accuracy, FMeasure, AUC] = evaluatePerformance(testData{:, end}, predictions, scores);
end

function rules = generateRulesFromTree(tree)
    % Generate rules from a decision tree
    % Arguments:
    % - tree: Trained decision tree
    % Returns:
    % - rules: Extracted rules
    
    % Get the paths from root to leaves
    [paths, nodeValues] = getTreePaths(tree);

    rules = {};
    for i = 1:length(paths)
        rule = struct();
        rule.conditions = paths{i};
        rule.class = nodeValues{i};
        rules{end+1} = rule;
    end
end

function [paths, nodeValues] = getTreePaths(tree)
    % Helper function to get paths from root to leaves in a decision tree
    % Arguments:
    % - tree: Trained decision tree
    % Returns:
    % - paths: Cell array of paths (conditions) from root to leaves
    % - nodeValues: Cell array of node values (class labels) for each path
    
    paths = {};
    nodeValues = {};
    function traverse(node, path)
        if isempty(tree.Children(node, 1)) && isempty(tree.Children(node, 2))
            paths{end+1} = path;
            nodeValues{end+1} = tree.NodeClass(node);
        else
            if ~isempty(tree.Children(node, 1))
                traverse(tree.Children(node, 1), [path, tree.CutPredictor(node), '<=', num2str(tree.CutPoint(node))]);
            end
            if ~isempty(tree.Children(node, 2))
                traverse(tree.Children(node, 2), [path, tree.CutPredictor(node), '>', num2str(tree.CutPoint(node))]);
            end
        end
    end
    traverse(1, {});
end

function prunedRules = pruneRules(rules, data)
    % Prune redundant rules
    % Arguments:
    % - rules: Generated rules
    % - data: Training dataset
    % Returns:
    % - prunedRules: Pruned rules
    
    prunedRules = rules;
    % rule pruning logic (e.g., removing redundant rules)
   
end

function [predictions, scores] = applyRules(rules, testData)
    % Apply rules to test data
    % Arguments:
    % - rules: Generated rules
    % - testData: Testing dataset
    % Returns:
    % - predictions: Predicted class labels
    % - scores: Confidence scores for the predictions
    
    numTestInstances = size(testData, 1);
    predictions = zeros(numTestInstances, 1);
    scores = zeros(numTestInstances, 1);

    for i = 1:numTestInstances
        instance = testData{i, 1:end-1};
        [predictions(i), scores(i)] = classifyInstance(rules, instance);
    end
end

function [label, score] = classifyInstance(rules, instance)
    % Classify a single instance using the generated rules
    % Arguments:
    % - rules: Generated rules
    % - instance: Instance to classify
    % Returns:
    % - label: Predicted class label
    % - score: Confidence score for the prediction

    for i = 1:length(rules)
        if satisfiesConditions(rules{i}.conditions, instance)
            label = rules{i}.class;
            score = 1;  % Placeholder for confidence score
            return;
        end
    end
    label = NaN;  % Default label if no rule applies
    score = 0;  % Default score if no rule applies
end

function satisfied = satisfiesConditions(conditions, instance)
    % Check if an instance satisfies a set of conditions
    % Arguments:
    % - conditions: Set of conditions (e.g., path from tree)
    % - instance: Instance to check
    % Returns:
    % - satisfied: Boolean indicating whether the instance satisfies the conditions
    
    satisfied = true;
    for i = 1:2:length(conditions)
        predictor = conditions{i};
        operator = conditions{i+1};
        value = str2double(conditions{i+2});
        if strcmp(operator, '<=')
            if instance.(predictor) > value
                satisfied = false;
                break;
            end
        elseif strcmp(operator, '>')
            if instance.(predictor) <= value
                satisfied = false;
                break;
            end
        end
    end
end

function [accuracy, FMeasure, AUC] = evaluatePerformance(trueLabels, predictions, scores)
    % Evaluate the performance of the classifier
    % Compute accuracy, F-Measure, and AUC
    % Arguments:
    % - trueLabels: True class labels
    % - predictions: Predicted class labels
    % - scores: Confidence scores for the predictions
    % Returns:
    % - accuracy: Classification accuracy
    % - FMeasure: F-measure of the classifier
    % - AUC: Area Under the ROC Curve

    accuracy = sum(trueLabels == predictions) / length(trueLabels);
    precision = sum((trueLabels == 1) & (predictions == 1)) / sum(predictions == 1);
    recall = sum((trueLabels == 1) & (predictions == 1)) / sum(trueLabels == 1);
    FMeasure = 2 * (precision * recall) / (precision + recall);
    [X, Y, ~, AUC] = perfcurve(trueLabels, scores, 1);
end

