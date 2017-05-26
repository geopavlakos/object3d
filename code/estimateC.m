function C = estimateC(W,R,B,C,lam,tol)

% INPUT:
% W: 2-by-P matrix
% R: 2-by-3 matrix
% B: 3K-by-P matrx
% C: 1-by-K vector

% This function estimates c by minimizing
% f(C) + lam*r(C), where
% f(C) = 0.5 * norm(W-R*S,'fro')^2 where S = \sum_i C_i*B_i
% r(C) = \|C\|_1
% It implements proximal gradient + nesterov

P = size(W,2);
K = size(B,1)/3;

C = C'; % transpose to be a column vector
C0 = C; % C0: the previous estimate
t = 1; % auxiliary variable for nesterov
t0 = 1; % auxiliary variable for nesterov
fvalue = inf;

% next we work on the linear system y = X*C
y = W(:); % vectorized W
X = zeros(2*P,K); % each column is a rotated Bk
for k = 1:K
    RBk = R*B(3*k-2:3*k,:);
    X(:,k) = RBk(:);
end

if lam == 0
    C = X\y;
    C = C';
    return
end

% mu is set as the 2-norm of the Hessian of f(C)
mu = norm(X'*X); 

for iter = 1:1000
    
    % Z is an auxiliary variable in nesterov method
    Z = C + (t0-1)/t*(C-C0);
    
    % gradient descent 
    Z = Z + X'*(y-X*Z)/mu;
    
    % nonegative thresholding
%     Z = Z - lam/mu;
%     Z = max(Z,0);
    
    % soft thresholding
    Z = sign(Z).*max(abs(Z)-lam/mu,0);
    
    % update C
    C0 = C;
    C = Z;
    
    % update t
    t0 = t;
    t = (1+sqrt(1+4*t^2))/2;
    
    % function value
    fvalue0 = fvalue;
    fvalue = 0.5 * norm(y-X*C,'fro')^2 + lam*sum(abs(C));
    if fvalue > fvalue0
        t0 = 1; % APG with restart
        t = 1;
    end
    
    % check convergence
    RelChg = norm(C-C0,'fro')/(norm(C0,'fro')+eps) ;
%     fprintf('Iter %d: FunVal = %f, RelChg = %f\n',iter,fvalue,RelChg);
    if RelChg < tol
        break
    end
    
end

C = C'; % transpose back to be a row vector
