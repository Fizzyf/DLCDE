function adjacency_matrix = generate_KNN_graph(distances, K)
    % distances: 样本欧式距离矩阵，大小为 n×n
    % K: KNN 中的 K 值

    n = size(distances, 1);
    adjacency_matrix = zeros(n, n);

    % 对于每个样本，找到其最近的 K 个邻居
    for i = 1:n
        % 获取第 i 个样本到所有其他样本的距离
        dist_i = distances(:, i);
        
        % 对距离向量进行排序，获取前 K+1 个最小的距离的索引，
        % 因为最小的距离是样本自身，所以需要从第2个开始
        [~, sorted_indices] = sort(dist_i);
        k_nearest_indices = sorted_indices(2:K+1);
        
        % 在邻接矩阵中标记第 i 个样本和其 K 个最近邻的连接
        adjacency_matrix(i, k_nearest_indices) = 1;
    end

    % 将邻接矩阵设置为对称矩阵，因为KNN图是无向图
    adjacency_matrix = max(adjacency_matrix, adjacency_matrix');

    % 如果需要，你可以进一步归一化邻接矩阵
    % adjacency_matrix = adjacency_matrix ./ sum(adjacency_matrix, 2);
end


