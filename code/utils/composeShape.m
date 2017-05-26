function S = composeShape(B,C)

if size(C,2) ~= size(B,1)/3
    C = C';
end

f = size(C,1);
p = size(B,2);
k = size(B,1)/3;

B = reshape(B',3*p,k);
S = B*C';
S = reshape(S,p,3*f)';