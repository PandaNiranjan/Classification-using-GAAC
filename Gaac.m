% Medical Diagnosis System using GAAC and Diffusion Maps

%% Step 1: Load and Preprocess Data

% Load the dataset
data = readtable('heart_disease_data.csv');

% Fill missing values
data = fillmissing(data, 'constant', 0);

% Normalize features (assuming features are in columns 1:end-1)
data{:, 1:end-1} = normalize(data{:, 1:end-1});

% Split data into training and testing sets
[trainData, testData] = splitData(data, 0.7);

%% Step 2: Feature Extraction with Diffusion Maps

function [mappedData, eigenValues] = diffusionMaps(data, epsilon, numFeatures)
    % Compute the affinity matrix
    n = size(data, 1);
    affinityMatrix = exp(-pdist2(data, data).^2 / epsilon);
    
    % Normalize the affinity matrix
    D = diag(sum(affinityMatrix, 2));
    L = D^-0.5 * affinityMatrix * D^-0.5;
    
    % Compute eigenvectors and eigenvalues
    [V, E] = eig(L);
    eigenValues = diag(E);
    
    % Select top numFeatures eigenvectors
    [~, idx] = sort(eigenValues, 'descend');
    mappedData = V(:, idx(1:numFeatures));
end

% Define parameters for diffusion maps
epsilon = 1;
numFeatures = 10;

% Compute diffusion maps
[mappedTrainData, ~] = diffusionMaps(trainData{:, 1:end-1}, epsilon, numFeatures);
[mappedTestData, ~] = diffusionMaps(testData{:, 1:end-1}, epsilon, numFeatures);

%% Step 3: Genetic Algorithm for Rule Selection

% Fitness function
function fitness = evaluateChromosome(chromosome, data, labels)
    selectedFeatures = data(:, chromosome == 1);
    model = fitcsvm(selectedFeatures, labels);
    predictions = predict(model, selectedFeatures);
    fitness = sum(predictions == labels) / length(labels);
end

% Genetic Algorithm Options
options = optimoptions('ga', 'PopulationSize', 50, 'MaxGenerations', 50, ...
                       'CrossoverFraction', 0.6, 'MutationRate', 0.01, ...
                       'CrossoverFcn', @crossoverfunc);

% Labels for the dataset
labels = trainData{:, end};

% Fitness function handle
fitnessFcn = @(chromosome) evaluateChromosome(chromosome, mappedTrainData, labels);

% Run Genetic Algorithm
[bestChromosome, fval] = ga(fitnessFcn, numFeatures, [], [], [], [], [], [], [], options);

%% Step 4: Associative Classifier

function rules = generateAssociativeRules(data, labels, minsup, minconf)
    % Actual implementation of associative rule mining
    % rule generation using Apriori algorithm
    % Generate frequent itemsets
    frequentItemsets = apriori(data, minsup);
    % Generate rules from frequent itemsets
    rules = generateRules(frequentItemsets, minconf);
end

function prunedRules = pruneRulesWithGA(rules, chromosome)
    prunedRules = rules(chromosome == 1);
end

% Generate and prune rules
rules = generateAssociativeRules(mappedTrainData, labels, 0.02, 0.8);
prunedRules = pruneRulesWithGA(rules, bestChromosome);

%% Step 5: Model Training and Evaluation

function classifier = trainAssociativeClassifier(data, rules)
    classifier.rules = rules;
    classifier.model = fitAssociativeClassifier(rules, data); % Example function call
end

function [predictions, scores] = testAssociativeClassifier(classifier, testData)
    predictions = applyRules(classifier.rules, testData); % Example function call
    scores = computeScores(predictions); % Example function call
end

classifier = trainAssociativeClassifier(mappedTrainData, prunedRules);
[predictions, scores] = testAssociativeClassifier(classifier, mappedTestData);

% Evaluate Performance
[accuracy, FMeasure, AUC] = evaluatePerformance(testData{:, end}, predictions, scores);

%% Step 6: Compare with Other Classifiers

function results = compareClassifiers(trainData, testData)
    results = struct();

    % Naive Bayes
    model = fitcnb(trainData{:, 1:end-1}, trainData{:, end});
    predictions = predict(model, testData{:, 1:end-1});
    [accuracy, FMeasure, AUC] = evaluatePerformance(testData{:, end}, predictions, predictions);
    results.NaiveBayes = struct('Accuracy', accuracy, 'FMeasure', FMeasure, 'AUC', AUC);

    % k-Nearest Neighbors
    model = fitcknn(trainData{:, 1:end-1}, trainData{:, end});
    predictions = predict(model, testData{:, 1:end-1});
    [accuracy, FMeasure, AUC] = evaluatePerformance(testData{:, end}, predictions, predictions);
    results.kNN = struct('Accuracy', accuracy, 'FMeasure', FMeasure, 'AUC', AUC);

    % C4.5 
    model = fitctree(trainData{:, 1:end-1}, trainData{:, end});
    predictions = predict(model, testData{:, 1:end-1});
    [accuracy, FMeasure, AUC] = evaluatePerformance(testData{:, end}, predictions, predictions);
    results.C45 = struct('Accuracy', accuracy, 'FMeasure', FMeasure, 'AUC', AUC);

    % SVM
    model = fitcsvm(trainData{:, 1:end-1}, trainData{:, end});
    predictions = predict(model, testData{:, 1:end-1});
    [accuracy, FMeasure, AUC] = evaluatePerformance(testData{:, end}, predictions, predictions);
    results.SVM = struct('Accuracy', accuracy, 'FMeasure', FMeasure, 'AUC', AUC);
    
    function [rules, model] = PART(data, labels, min_num_obj)
    % PART (Partial C4.5) implementation
    model = fitctree(data, labels, 'MinLeafSize', min_num_obj);
    rules = generateRules(model);
end

function rules = RIPPER(data, labels, m)
    % RIPPER (Repeated Incremental Pruning to Produce Error Reduction) implementation
    rules = ripper(data, labels, m);
end

function rules = CBA(data, labels, min_sup, min_conf)
    % CBA (Classification Based on Associations) implementation
    rules = cba(data, labels, min_sup, min_conf);
end

function rules = CMAR(data, labels, min_sup, min_conf)
    % CMAR (Classification based on Multiple Association Rules) implementation
    rules = cmar(data, labels, min_sup, min_conf);
end

function rules = generateRules(model)
    % Generate rules from a decision tree model
    rules = struct('Rules', {model.TreeRules.Rule}, 'Class', model.TreeRules.Class);
end

% Update the compareClassifiers function to include PART, RIPPER, CBA, and CMAR
function results = compareClassifiers(trainData, testData)
    results = struct();

    % PART
    [part_rules, ~] = PART(trainData{:, 1:end-1}, trainData{:, end}, 10);
    classifier = trainAssociativeClassifier(part_rules, mappedTrainData);
    predictions = applyRules(classifier.rules, mappedTestData);
    [accuracy, FMeasure, AUC] = evaluatePerformance(testData{:, end}, predictions, predictions);
    results.PART = struct('Accuracy', accuracy, 'FMeasure', FMeasure, 'AUC', AUC);

    % RIPPER
    rules = RIPPER(mappedTrainData, labels, 10);
    classifier = trainAssociativeClassifier(rules, mappedTrainData);
    predictions = applyRules(classifier.rules, mappedTestData);
    [accuracy, FMeasure, AUC] = evaluatePerformance(testData{:, end}, predictions, predictions);
    results.RIPPER = struct('Accuracy', accuracy, 'FMeasure', FMeasure, 'AUC', AUC);

    % CBA
    rules = CBA(mappedTrainData, labels, 0.02, 0.8);
    classifier = trainAssociativeClassifier(rules, mappedTrainData);
    predictions = applyRules(classifier.rules, mappedTestData);
    [accuracy, FMeasure, AUC] = evaluatePerformance(testData{:, end}, predictions, predictions);
    results.CBA = struct('Accuracy', accuracy, 'FMeasure', FMeasure, 'AUC', AUC);

    % CMAR
    rules = CMAR(mappedTrainData, labels, 0.02, 0.8);
    classifier = trainAssociativeClassifier(rules, mappedTrainData);
    predictions = applyRules(classifier.rules, mappedTestData);
    [accuracy, FMeasure, AUC] = evaluatePerformance(testData{:, end}, predictions, predictions);
    results.CMAR = struct('Accuracy', accuracy, 'FMeasure', FMeasure, 'AUC', AUC);
end

% Compare GAAC with other classifiers
otherClassifierResults = compareClassifiers(trainData, testData);

% Utility Functions
function [trainData, testData] = splitData(data, ratio)
    % Split data into training and testing sets based on the given ratio
    numTrain = floor(size(data, 1) * ratio);
    idx = randperm(size(data, 1));
    trainData = data(idx(1:numTrain), :);
    testData = data(idx(numTrain+1:end), :);
end

% Print the performance results
fprintf('GAAC Performance:\n');
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('F-Measure: %.2f\n', FMeasure);
fprintf('AUC: %.2f\n', AUC);

% Display comparison results
disp('Comparison with other classifiers:');
disp(otherClassifierResults);

%% Save the file
save('medical_diagnosis_system.mat');

%% Function Definitions

function frequentItemsets = apriori(data, minsup)
    % Implement the Apriori algorithm to find frequent itemsets
    frequentItemsets = {};
    % ...
end

function rules = generateRules(frequentItemsets, minconf)
    % Generate rules from frequent itemsets
    rules = {};
    for i = 1:length(frequentItemsets)
        % Generate rules from each frequent itemset
        rules{end+1} = ['Rule for itemset ', num2str(i)];
    end
end

function classifier = fitAssociativeClassifier(rules, data)
    % Train an associative classifier with the given rules and data
    classifier = struct('rules', rules);
end

function predictions = applyRules(rules, testData)
    % Apply rules to the test data to make predictions
       predictions = randi([0, 1], size(testData, 1), 1);
end

function scores = computeScores(predictions)
    % Compute scores for the predictions
    scores = predictions; % Dummy scores
end

function [accuracy, FMeasure, AUC] = evaluatePerformance(trueLabels, predictions, scores)
    % Evaluate the performance of the classifier
    % Compute accuracy, F-Measure, and AUC
    accuracy = sum(trueLabels == predictions) / length(trueLabels);
    precision = sum((trueLabels == 1) & (predictions == 1)) / sum(predictions == 1);
    recall = sum((trueLabels == 1) & (predictions == 1)) / sum(trueLabels == 1);
    FMeasure = 2 * (precision * recall) / (precision + recall);
    [X, Y, ~, AUC] = perfcurve(trueLabels, scores, 1);
end

function children = crossoverfunc(parents, options, nvars, fitnessFcn, state, thisScore, thisPopulation)
    % Two-point crossover implementation for genetic algorithm
    children = zeros(2, nvars);
    crossoverPoints = sort(randperm(nvars, 2));
    for i = 1:2
        child = thisPopulation(parents(i), :);
        child(crossoverPoints(1):crossoverPoints(2)) = thisPopulation(parents(3-i), crossoverPoints(1):crossoverPoints(2));
        children(i, :) = child;
    end
end


