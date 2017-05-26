function C = estimateC_weighted(W,R,B,D,lam)

P = size(W,2);
K = size(B,1)/3;

d = diag(D);
D = zeros(2*P,2*P);
for i = 1:P
    D(2*i-1,2*i-1) = d(i);
    D(2*i,2*i) = d(i);
end

% next we work on the linear system y = X*C
y = W(:); % vectorized W
X = zeros(2*P,K); % each column is a rotated Bk
for k = 1:K
    RBk = R*B(3*k-2:3*k,:);
    X(:,k) = RBk(:);
end

C = pinv(X'*D*X+lam*eye(size(X,2)))*X'*D*y;
C = C';