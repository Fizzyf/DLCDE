function EuD = EUD(X)
    % 获取特征矩阵的维度信息
    [d, n] = size(X);
    
    % 初始化欧氏距离矩阵
    EuD = zeros(n, n);
    
    % 计算欧氏距离
    for i = 1:n
        for j = i+1:n
            % 计算第i个样本与第j个样本之间的欧氏距离
            distance = norm(X(:, i) - X(:, j));
            
            % 填充欧氏距离矩阵，由于欧氏距离是对称的，因此同时填充(i, j)和(j, i)位置
            EuD(i, j) = distance;
            EuD(j, i) = distance;
        end
    end
end


