% Ref: A. Del Bue, J. Xavier, L. Agapito, and M. Paladini, "Bilinear
% Factorization via Augmented Lagrange Multipliers (BALM)" ECCV 2010.
%
%  This program is free software; you can redistribute it and/or
%  modify it under the terms of the GNU General Public License
%  as published by the Free Software Foundation; version 2, June 1991 
%
% USAGE: Y = proj_deformable_approx(X)
%
% This function projects a generic matrix X of size 3*K x 2 where K is the 
% number of basis shapes into the matrix Y that satisfy the manifold
% constraints. This projection is an approximation of the projector
% introduced in: M. Paladini, A. Del Bue, S. M. s, M. Dodig, J. Xavier, and
% L. Agapito, "Factorization for Non-Rigid and Articulated Structure using
% Metric Projections" CVPR 2009. Check the BALM paper, Sec 5.1.
%
% INPUT
%
% X: the 3*K x 2 affine matrix 
%
% OUTPUT
%
% Y: the 3*K x 2 with manifold constraints

function [Y,L,Q] = proj_deformable_approx(X)

if nargin < 1
    
    Y = 2; % this says on how many columns of M the projection must be done
    return
end
    
r = size(X,1);
d = r/3;

A = zeros(3,3);
for i = 1:d
    Ai = X((i-1)*3+1:i*3,:);
    A = A + Ai*Ai';
end;

[U,S,V] = svd(A);

Q = U(:,1:2);

G = zeros(2,2);
for i = 1:d
    Ai = X((i-1)*3+1:i*3,:);
    Ti = Q'*Ai;
    gi = [ trace(Ti) ; Ti(2,1)-Ti(1,2) ];
    G = G + gi*gi';
end;

[U1,S1,V1] = svd(G);

G = zeros(2,2);
for i = 1:d
    Ai = X((i-1)*3+1:i*3,:);
    Ti = Q'*Ai;
    gi = [ Ti(1,1)-Ti(2,2) ; Ti(1,2)+Ti(2,1) ];
    G = G + gi*gi';
end;

[U2,S2,V2] = svd(G);

if S1(1,1) > S2(1,1)
    u = U1(:,1);
    R = [ u(1) -u(2) ; u(2) u(1) ];
else
    u = U2(:,1);
    R = [ u(1) u(2) ; u(2) -u(1) ];
end;

Q = Q*R;

Y = [];
L = [];
for i = 1:d
    Ai = X((i-1)*3+1:i*3,:);
    ti = 0.5*trace(Q'*Ai);
%     if i == 1 && ti < 0
%         ti = -ti;
%         Q = -Q;
%     end
    L = [ L ; ti];
    Y = [ Y ; ti*Q ];
end;