function S = reshapeS(S,mode)

switch mode
    case 'b2v'
        [F,P] = size(S);
        F = F/3;
        S = reshape(S',3*P,F);
    case 'v2b'
        [P,F] = size(S);
        P = P/3;
        S = reshape(S,P,3*F)';
end