LUT = zeros(65, 32);

for x = 0 : 64
    P = zeros(32,32);
    for i = 1 : 31
        P(i, i + 1) = x / 64;
        P(i + 1, i) = 1 - x / 64;
    end

    P(1, 1) = 1 - x / 64;
    P(32, 32) = x / 64;

    [V,D] = eig(P'); % Find eigenvalues and left eigenvectors of A
    [~,ix] = min(abs(diag(D)-1)); % Locate an eigenvalue which equals 1
    v = V(:,ix)'; % The corresponding row of V' will be a solution
    LUT(x + 1, :) = v/sum(v); % Adjust it to have a sum of 1
end

LUT = round(LUT .* 64);

filename = '32_state_fsm_steady_state_distribution.xlsx';
writematrix(LUT,filename,'Sheet',1)