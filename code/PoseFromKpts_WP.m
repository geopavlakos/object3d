function output = PoseFromKpts_WP(W,dict,varargin)

% data size
B = dict.mu;
pc = dict.pc;

[k,p] = size(B);
k = k/3;

lam = 1;
alpha = 1;
D = eye(p);
tol = 1e-3;
verb = true;

ivargin = 1;
while ivargin <= length(varargin)
    switch lower(varargin{ivargin})
        case 'lam'
            ivargin = ivargin + 1;
            lam = diag(varargin{ivargin});
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
B = bsxfun(@minus,B,mean(B,2));

% initialization
M = zeros(2,3*k); 
C = zeros(1,k); % norm of each Xi

% auxiliary variables for ADMM
Z = M;
Y = M;
mu = 1/(mean(abs(W(:)))+eps);
% pre-computing
BBt = B*D*B';

t0 = tic;
for iter = 1:1000
    
    % update translation
    T = sum((W-Z*B)*D,2)/(sum(diag(D))+eps);
    W2fit = W - T*ones(1,p);
    
    % update motion matrix Z
    Z0 = Z;
    Z = (W2fit*D*B'+mu*M+Y)/(BBt+mu*eye(3*k));
    
    % update motion matrix M
    Q = Z - Y/mu;
    for i = 1:k
        [M(:,3*i-2:3*i),C(i)] = prox_2norm(Q(:,3*i-2:3*i),alpha/mu);
    end

    % update dual variable
    Y = Y + mu*(M-Z);
    
    PrimRes = norm(M-Z,'fro')/(norm(Z0,'fro')+eps);
    DualRes = mu*norm(Z-Z0,'fro')/(norm(Z0,'fro')+eps);
    
    % show output
    if verb %&& mod(iter,10) == 0
        fprintf('Iter %d: PrimRes = %f, DualRes = %f, mu = %f\n',...
            iter,PrimRes,DualRes,mu);
    end
    
    % Convergent? 
    if  PrimRes < tol && DualRes < tol
        break
    else
        if PrimRes>10*DualRes
            mu = 2*mu;
        elseif DualRes>10*PrimRes
            mu = mu/2;
        else
        end
    end
end
t1 = toc(t0);
[R,C] = syncRot(M);
if sum(abs(R(:))) == 0
    R = eye(3);
end

R = R(1:2,:);
S = kron(C,eye(3))*B;

%%
fval = inf;
t0 = tic;
C = zeros(1,size(pc,1)/3);
for iter = 1:1000
    
    % update translation 
    T = sum((W-R*S)*D,2)/(sum(diag(D))+eps);
    W2fit = W - T*ones(1,p);
    
    % update rotation
    R = estimateR_weighted(S,W2fit,D,R);

    % update shape
    if isempty(pc)
        C0 = estimateC_weighted(W2fit,R,B,D,1e-3);
        S = C0*B;
    else
        C0 = estimateC_weighted(W2fit-R*kron(C,eye(3))*pc,R,B,D,1e-3);
        C = estimateC_weighted(W2fit-R*C0*B,R,pc,D,lam);
        S = C0*B + kron(C,eye(3))*pc;
    end

    fvaltm1 = fval;
    fval = 0.5*norm((W2fit-R*S)*sqrt(D),'fro')^2 + 0.5*norm(C)^2;
    
    % show output
    if verb %&& mod(iter,10) == 0
        fprintf('Iter: %d, fval = %f\n',iter,fval);
    end
    
    % check convergence
    if abs(fval-fvaltm1)/fvaltm1 < tol
        break
    end
    
end
t2 = toc(t0);

R(3,:) = cross(R(1,:),R(2,:));

%%
output.S = S;
output.M = M;
output.R = R;
output.C = C;
output.C0 = C0;
output.T = T;
output.fval = fval;
output.timeInit = t1;
output.timeAlt = t2;

end

function [X,normX] = prox_2norm(Z,lam)
% X is a 3-by-2 matrix
[U,W,V] = svd(Z,'econ');
w = diag(W);
if sum(w) <= lam
    w = [0,0];
elseif w(1)-w(2) <= lam
    w(1) = (sum(w)-lam)/2;
    w(2) = w(1);
else
    w(1) = w(1) - lam;
    w(2) = w(2);
end
X = U*diag(w)*V';
normX = w(1);
end