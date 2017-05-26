function R = findRotation(S1,S2)

[F,P] = size(S1);
F = F/3;

S1 = reshape(S1,3,F*P);
S2 = reshape(S2,3,F*P);

R = S1*S2';
[U,~,V] = svd(R);
R = U*V';
R = U*diag([1 1 det(R)])*V';