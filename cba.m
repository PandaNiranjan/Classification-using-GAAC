function rules = cba(train_data, train_labels, min_support, min_confidence)
    % Train_data: Training data (N x D matrix)
    % Train_labels: Training labels (N x 1 vector)
    % Min_support: Minimum support threshold for generating rules
    % Min_confidence: Minimum confidence threshold for generating rules
    
    % Convert the training data and labels into a table
    train_table = array2table([train_data, train_labels], 'VariableNames', ...
        [strsplit('Feature', num2str(size(train_data, 2))), 'Class']);
    
    % Find frequent itemsets using Apriori algorithm
    min_support_count = min_support * size(train_table, 1);
    itemsets = find_freq_itemsets(train_table, min_support_count);
    
    % Generate rules from frequent itemsets
    rules = generate_rules(itemsets, min_confidence, train_table);
end

function freq_itemsets = find_freq_itemsets(data_table, min_support_count)
    % Find frequent itemsets using Apriori algorithm
    itemsets = apriori(data_table(:, 1:end-1), 'MinSupport', min_support_count);
    
    % Convert itemsets to a cell array for easier manipulation
    freq_itemsets = cell(size(itemsets, 1), 1);
    for i = 1:size(itemsets, 1)
        freq_itemsets{i} = itemsets(i, :);
    end
end

function rules = generate_rules(freq_itemsets, min_confidence, data_table)
    rules = [];
    for i = 1:length(freq_itemsets)
        itemset = freq_itemsets{i};
        for j = 1:length(itemset)-1
            antecedent = itemset{j};
            consequent = itemset{end};
            confidence = calculate_confidence(antecedent, consequent, data_table);
            if confidence >= min_confidence
                rules = [rules; antecedent, consequent, confidence];
            end
        end
    end
end

function confidence = calculate_confidence(antecedent, consequent, data_table)
    antecedent_support = calculate_support(antecedent, data_table);
    rule_support = calculate_support([antecedent, consequent], data_table);
    confidence = rule_support / antecedent_support;
end

function support = calculate_support(itemset, data_table)
    % Calculate the support of an itemset in the data table
    support = sum(all(ismember(data_table(:, 1:end-1), itemset), 2)) / size(data_table, 1);
end

