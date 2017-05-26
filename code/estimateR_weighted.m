function R = estimateR_weighted(S,W,D,R0)

A = S';
B = W';
X0 = R0(1:2,:)';

warning('off', 'manopt:getHessian:approx');

[m,n,N] = size(A);
p = size(B,2);

At = zeros([n,m,N]);
for i=1:N
    At(:,:,i) =  A(:,:,i)';
end

manifold = stiefelfactory(n,p, N);
problem.M = manifold;

function [f,store] = cost(X,store)
    if ~isfield(store, 'E')
        store.E = multiprod(A,X) - B;
    end
    E = store.E;
    f = trace(E'*D*E)/(2*N);
end

% Riemannian gradient of the cost function.
function [g, store] = grad(X, store)

    if ~isfield(store, 'E')
        [~, store] = cost(X, store);
    end
    E = store.E;
    % Compute the Euclidean gradient of the cost wrt the rotations R
    % and wrt the cloud A,
    egrad = multiprod(At/N,D*E);
    % then transform this Euclidean gradient into the Riemannian
    % gradient.
    g = manifold.egrad2rgrad(X, egrad);
    store.egrad = egrad;
end


% Setup the problem structure with manifold M and cost+grad functions.

problem.cost = @cost;
problem.grad = @grad;

options.verbosity = 0;
options.tolgradnorm = 1e-3;
options.maxiter = 10;
X = trustregions(problem,X0,options);

R = X';

end