function [model_new,w,R,T] = fullShape(S1,model,kpt2fit)

if nargin < 3
    kpt2fit = 1:size(S1,2);
end

S2 = zeros(3,length(kpt2fit));
for i = 1:length(kpt2fit)
    xyz = model.(model.pnames{kpt2fit(i)});
    S2(:,i) = xyz';
end

% compare two structures and align S2 to S1 with similarity transformation

T1 = mean(S1,2);
S1 = bsxfun(@minus,S1,T1);
T2 = mean(S2,2);
S2 = bsxfun(@minus,S2,T2);
R = findRotation(S1,S2);
S2 = R*S2;
w = trace(S1'*S2)/(trace(S2'*S2)+eps);
T = T1 - w*R*T2;
vertices = bsxfun(@plus,w*R*bsxfun(@minus,(model.vertices)',T2),T1)';
model_new = model;
model_new.vertices = vertices;

end

function R = findRotation(S1,S2)
[F,P] = size(S1);
F = F/3;
S1 = reshape(S1,3,F*P);
S2 = reshape(S2,3,F*P);
R = S1*S2';
[U,~,V] = svd(R);
R = U*V';
R = U*diag([1 1 det(R)])*V';
end
