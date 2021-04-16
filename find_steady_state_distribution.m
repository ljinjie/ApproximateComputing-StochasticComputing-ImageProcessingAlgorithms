LUT = zeros(14, 16);

for x = 0 : 13
    P = zeros(16,16);
    for i = 1 : 15
        P(i, i + 1) = x / 13;
        P(i + 1, i) = 1 - x / 13;
    end

    P(1, 1) = 1 - x / 13;
    P(16, 16) = x / 13;

    [V,D] = eig(P'); % Find eigenvalues and left eigenvectors of A
    [~,ix] = min(abs(diag(D)-1)); % Locate an eigenvalue which equals 1
    v = V(:,ix)'; % The corresponding row of V' will be a solution
    LUT(x + 1, :) = v/sum(v); % Adjust it to have a sum of 1
end

LUT = round(LUT .* 32);

filename = '16_state_fsm_steady_state_distribution_14.xlsx';
writematrix(LUT,filename,'Sheet',1)