function S = KNNgraph(similarity_matrix, L, K)
    % 输入：
    % similarity_matrix: 样本对之间的相似度矩阵 (n×n)
    % L: 标签矩阵，表示每个样本对的类别标签 (c×n)
    % K: KNN的参数
    
    % 输出：
    % S: 类别相似度矩阵 (c×c)

    % 获取类别数量
    c = size(L, 1);

    % 初始化标签相似度矩阵S
    S = zeros(c, c);

    % 计算标签相似度
    for i = 1:c
        for j = i+1:c
            % 获取属于标签类别i和j的样本对的索引
            idx_i = find(L(i, :) == 1);
            idx_j = find(L(j, :) == 1);
            
            % 统计属于标签类别i和j的样本对在KNN图中连接到相同最近邻的数量
            count_same_neighbors = sum(sum(similarity_matrix(idx_i, idx_j)));

            % 计算标签相似度度量，可以根据任务需求选择不同的方法
            similarity = count_same_neighbors / K;  % 例如，这里简单地计算共同最近邻的比例

            % 将相似度放入S矩阵中
            S(i, j) = similarity;
            S(j, i) = similarity;  % S矩阵是对称的，需要同时更新两个位置
        end
    end
end


