function similarity_matrix = cosinesimilarity(X)
    % 计算样本之间的余弦相似度矩阵
    
    % 规范化样本矩阵，使其每列表示一个样本的特征向量
    X = X ./ sqrt(sum(X.^2, 1));
    
    % 计算余弦相似度
    similarity_matrix = X' * X;
end


