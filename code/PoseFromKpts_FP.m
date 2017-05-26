function output = PoseFromKpts_FP(W,dict,varargin)

% this function solves
% min ||W*diag(Z)-R*S-T||^2 + ||C||^2
% where S = C1*B1+...+Cn*Bn, 
% Z denotes the depth of points

lam = 1;
D = eye(size(W,2));
tol = 1e-3;
verb = true;
R = eye(3);

ivargin = 1;
while ivargin <= length(varargin)
    switch lower(varargin{ivargin})
        case 'r0'
            ivargin = ivargin + 1;
            R = varargin{ivargin};
        case 'lam'
            ivargin = ivargin + 1;
            lam = varargin{ivargin};
        case 'weight'
            ivargin = ivargin + 1;
            D = diag(varargin{ivargin});
        case 'tol'
            ivargin = ivargin + 1;
            tol = varargin{ivargin};
        case 'verb'
            ivargin = ivargin + 1;
            verb = varargin{ivargin};
        otherwise
            fprintf('Unknown option ''%s'' is ignored !\n',varargin{ivargin});
    end
    ivargin = ivargin + 1;
end

% centralize basis
mu = centralize(dict.mu);
pc = centralize(dict.pc);

% initialization
S = mu;
T = mean(W,2)*mean(std(R(1:2,:)*S,1,2))/(mean(std(W,1,2))+eps);
C = 0;

fval = inf;
t0 = tic;
for iter = 1:1000
    
    % update dpeth Z
    Z = sum(W.*bsxfun(@plus,R*S,T),1)./(sum(W.^2,1)+eps);
    
    % update R and T by aligning S to W*diag(Z);
    Sp = W*diag(Z);
    T = sum((Sp-R*S)*D,2)/(sum(diag(D))+eps);
    St = bsxfun(@minus,Sp,T);
    [U,~,V] = svd(St*D*S'); 
    R = U*diag([1,1,sign(det(U*V'))])*V';

    % update C by least squares
    if ~isempty(pc)
        y = reshapeS(R'*St-mu,'b2v');
        X = reshapeS(pc,'b2v');
        C = pinv(X'*X+lam*eye(length(C)))*X'*y;
        S = mu + composeShape(pc,C);
    end

    fvaltm1 = fval;
    fval = norm((St-R*S)*sqrt(D),'fro')^2 + lam*norm(C)^2;
    
    % show output
    if verb
        fprintf('Iter: %d, fval = %f\n',iter,fval);
    end
    
    % check convergence
    if abs(fval-fvaltm1)/(fvaltm1+eps) < tol
        break
    end

end

output.S = S;
output.R = R;
output.C = C;
output.T = T;
output.Z = Z;
output.time = toc(t0);

