function affinity_matrix = knn_to_affinity(knn_graph, distances)
    % knn_graph: KNN图的邻接矩阵
    % distances: 样本欧式距离矩阵

    n = size(knn_graph, 1);
    affinity_matrix = zeros(n, n);

    % 计算每个样本到其 KNN 邻居的平均距离
    avg_distances = zeros(n, 1);
    for i = 1:n
        % 获取第 i 个样本的 KNN 邻居索引
        knn_indices = find(knn_graph(i, :));
        
        % 计算第 i 个样本到其 KNN 邻居的平均距离
        avg_distances(i) = mean(distances(i, knn_indices));
    end

    % 使用平均距离来估算 sigma（可以按需要调整常数因子）
    sigma = 0.1 * mean(avg_distances);

    % 重新遍历 KNN 图，计算亲和度
    for i = 1:n
        for j = 1:n
            % 如果样本 i 和样本 j 有连接，计算亲和度
            if knn_graph(i, j) == 1
                % 使用高斯核函数计算亲和度，使用自动估算的 sigma
                affinity_matrix(i, j) = exp(-distances(i, j)^2 / (2 * sigma^2));
            end
        end
    end

    % 如果需要，可以进行归一化，使亲和度矩阵满足对称性和归一性的要求
    % affinity_matrix = affinity_matrix + affinity_matrix';
    % affinity_matrix = affinity_matrix / max(max(affinity_matrix));
end





