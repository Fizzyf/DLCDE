function [B] = train_DLCDE(L1,XKTrain,YKTrain,LTrain,param)
    % parameters
    nbits = param.nbits;
    n = size(LTrain,1);
    c = size(LTrain,2);

    % initization
    B = sign(randn(nbits, n)); B(B==0) = -1;
    V = randn(nbits, n);
    W = randn(c,nbits);
    Z1 = eye(c);
    Uy = randn(size(YKTrain,2),nbits); 
    S = L1*L1';
    X = XKTrain';
    Y = YKTrain';
    L = LTrain';
    L1 = L1'; 
    C = L1*L1';


    %%iteration start
    for i = 1:param.max_iter
        fprintf('iteration %3d\n', i);
        
        % update U
        temp1 = X*V'; temp2 = Y*V';
        Ux = (param.alpha1*temp1 - param.omega*Uy)/(n*eye(nbits));
        Uy = (param.alpha2*temp2 - param.omega*Ux)/(n*eye(nbits));
        Ul = (param.lambda*L*V')/(n*eye(nbits));  %BATCH's label matrix factorization, kind of useless
        clear temp1 temp2
        
        % update V
        Z = nbits*(B*S) + param.beta*B + param.alpha1*(Ux'*X)...
             + param.alpha2*(Uy'*Y) + param.lambda*(Ul'*L); %we dont have label as first modal like BATCH
        %Temp = Z - 1/n*ones(n,1)*(ones(1,n)*Z);
        %[P,Lmd,Q] = svd(Temp);
        %idx = (diag(Lmd)>1e-6);
        %Q = Q(:,idx); Q_ = orth(Q(:,~idx));
        %P = P(:,idx); P_ = orth(P(:,~idx));
        %V = sqrt(n)*[P P_]*[Q Q_]';
        Temp = Z*Z'-1/n*(Z*ones(n,1)*(ones(1,n)*Z'));
        [~,Lmd,QQ] = svd(Temp); clear Temp
        idx = (diag(Lmd)>1e-6);
        Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
        P = (Z'-1/n*ones(n,1)*(ones(1,n)*Z')) *  (Q / (sqrt(Lmd(idx,idx))));
        P_ = orth(randn(n,nbits-length(find(idx==1))));
        V = sqrt(n)*[Q Q_]*[P P_]';
        
        %update W
        W = (Z1*L*B')/(B*B');

        %update Z1
        temp1 = ((param.gamma + param.miu)*eye(c)) + param.ksi*C;
        temp2 = param.gamma*(L*L') + param.miu*W*B*L';
        Z1 = temp1\temp2/(L*L');
        clear temp1 temp2


        % update B
        B = sign(nbits*(V*S)+param.beta*V+param.miu*W'*Z1*L);
        

    end

    final_B = sign(B);
    final_B(final_B==0) = -1;
    B = final_B;

end