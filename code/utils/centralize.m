function S = centralize(S)

S = bsxfun(@minus,S,mean(S,2));
