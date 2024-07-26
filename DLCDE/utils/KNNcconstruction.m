function class_similarity_matrix = KNNcconstruction(L, X, Y, K)
    % 获取类别的数量
    c = size(L, 1);

    % 初始化类别相似度矩阵
    class_similarity_matrix = zeros(c, c);

    % 获取特征维度和样本数量
    [d_x, n] = size(X);
    [d_y, ~] = size(Y);

    % 遍历每一对不同的类别
    for i = 1:c
        for j = (i+1):c  % 仅遍历与i不同的类别
            % 找到属于类别i的样本的索引
            indices_i = find(L(i, :) == 1);

            % 找到属于类别j的样本的索引
            indices_j = find(L(j, :) == 1);

            % 从两个模态的特征矩阵中提取属于类别i和类别j的样本
            X_i = X(:, indices_i);
            Y_i = Y(:, indices_i);
            X_j = X(:, indices_j);
            Y_j = Y(:, indices_j);

            % 合并两个模态的特征
            combined_features_i = [X_i; Y_i];
            combined_features_j = [X_j; Y_j];

            % 训练KNN模型
            knn_model = fitcknn(combined_features_i', ones(1, length(indices_i)), 'NumNeighbors', K);

            % 预测类别
            predicted_labels = predict(knn_model, combined_features_j');

            % 统计属于类别i的样本中有多少被预测为类别j的
            count = sum(predicted_labels == 1);

            % 计算相似度并存储在类别相似度矩阵中
            class_similarity_matrix(i, j) = count / length(indices_i);
            
            % 因为相似度矩阵是对称的，所以可以同时更新class_similarity_matrix(j, i)
            class_similarity_matrix(j, i) = class_similarity_matrix(i, j);
        end
    end
end





