function [h1 h0] = biorth_kernel(k0,k1)
% this function computes graph-Dk biorthogonal filters which have
% k0 zeros at the lam =0 for low-pass filter and k1 zeros at lam = 2 for
% high-pass filters and are maximally balanced.
%% P(\lambda) design
K = k1 + k0;
a = zeros(1, 2*K);
for i = 1:(K + 1)
    a(i) = nchoosek(K,i-1);
end
A = zeros(2*K);
for i = 1:2*K
    A(i,1:i) = fliplr(a(1:i));
end
A = A(1:2:end, 1:K);
b0 = zeros(K,1);
b0(1) = 1;
b = A\b0;
b = fliplr(b');
q = @(x)((1+x).^K.*polyval(b,x));
r = roots(b);
s =-1./r;
[dontcare index] = sort(abs(imag(s)));
s = s(index);

%% Filter Design
% In our proposed design we propose maximally balanced filters which
% are nearly orthogonal. we assume that at lam = -1,
% we assign k0 zeros to h0(1-lam) and k1 zeros to h1(1+lam). The
% the residual polynomial R(lam) has degree (k0 + k1 -1), and its
% zeros are divided so that h1 is always of degree (k1 + k0) and h0 is
% always of degree (k1 + k0 - 1); This gives a maximally balanced filter.
% The complex pair of roots of R(\lambda) are paired up with other
% complex pair of roots closest in magnitude. We design abs((k1 - k0 -1)/2)
% such pairs and divide each pair equally to h_1 and h_0 respectively. This
% is done as follows:

% 1. If (k1 + k0) = 4*m + 2, then the residual polynomial R(lam)
% has odd degree 4*m + 1, with exactly 1 real root and 4*m complex roots
% which occur as conjugate pairs. If k0 is odd, then we assign the
% real root and k0/2 complex conjgate pairs of roots to h1(lam) and
% remaining 2m -(k0 -1)/2
% complex conjgate pairs of roots to h_0. As a result h_1 is of length (k_1
% + k_0 ) and h_0 is of length (k-1 + k_0 -1).

% 2. If (k_1 + k_0) = 4*m  then the residual polynomial R(\lambda) has odd
% degree (4*m - 1) with exactly one real root and (4*m -2) complex roots
% which occur as conjugate pairs. If In this case, we assign the
% real root and 2*k-2 complex conjgate roots to h_0(\lambda) and  remaining 2*k
% complex conjgate roots to h_1. As a result h_1 is of length K and h_0
% is of length K-1.

% 3. If K = 4*k +1  then we assign 2*k number of zeros to h_0(\lambda) and
% 2*k+1 zeros to h_1(\lambda); The residual polynomial r(\lambda)
% has even degree 4*k and has 4*k complex roots
% which occur as conjugate pairs. In this case, we assign the
% 2*k complex conjgate roots each to h_0(\lambda) and  h_1 respectively.
% As a result h_1 is of length K and h_0 is of length K-1.

% 4. If K = 4*k + 3  then we assign 2*k +1 number of zeros to h_0(\lambda) and
% 2*k+2 zeros to h_1(\lambda); The residual polynomial r(\lambda)
% has even degree 4*k +2 and has 4*k+2 complex roots
% which occur in conjugate pairs. In this case, we assign the
% 2*k+1 complex conjgate roots each to h_0(\lambda) and  h_1 respectively.
% As a result h_1 is of length K and h_0 is of length K-1.

switch mod(K,2)
    case 0
        if mod(k1,2) == 0
            last = min([(2*k1 - 6), K-1]);
            last = max([last 0]);
            index0 = 2:4:last;
            index0 = sort([index0,(index0+1)]);
            index0 = [1 index0];
            index1 = setdiff(1:(K-1),index0);
        else
            last = min([(2*k1 - 4), K-1]);
            last = max([last 0]);
            index0 = 2:4:last;
            index0 = sort([index0,(index0+1)]);
            index1 = setdiff(1:(K-1),index0);
        end
    case 1
        if mod(k1,2) == 0 % in this case we make h_0 to be of bigger length than h_1
            last = min([(2*k1 - 3), K-1]);
            last = max([last 0]);
            index0 = 1:4:last;
            index0 = sort([index0,(index0+1)]);
            index1 = setdiff(1:(K-1),index0);
        else
            last = min([(2*k1 - 5), K-1]);
            last = max([last 0]);
            index0 = 1:4:last;
            index0 = sort([index0,(index0+1)]);
            index1 = setdiff(1:(K-1),index0);
        end
end
s1 = s(index1);
s0 = s(index0);
h1 = zeros(1,K+1);
h1(K-k1+1) = 1;
for i = 1:length(s1)
    h1 = conv(h1,[s1(i) (1-s1(i))]);
end
h1 = real(h1);
h0 = [-1 2];
for i = 1:(k0-1)
    h0 = conv(h0,[-1 2]);
end
for i = 1:length(s0)
    h0 = conv(h0,[-s0(i) (1+s0(i))]);
end
h0 = real(h0);

% Lam = -1:0.01:1;
% H1 = polyval(h1,1 + Lam);
% H0 = polyval(h0,1 - Lam);
% figure,
% plot(Lam,H1.*H0,'*')
% figure,
% plot(Lam,q(Lam),'r*')


%
%
%
%
% % if len is
