function [S,a,t] = normalizeS(S)

t = mean(S,2);
S = S - t*ones(1,size(S,2));
a = mean(std(S, 1, 2));
S = S / a;
