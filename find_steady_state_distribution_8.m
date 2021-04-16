LUT = zeros(17, 8);

for x = 0 : 16
    P = zeros(8,8);
    for i = 1 : 7
        P(i, i + 1) = x / 16;
        P(i + 1, i) = 1 - x / 16;
    end

    P(1, 1) = 1 - x / 16;
    P(8, 8) = x / 16;

    [V,D] = eig(P'); % Find eigenvalues and left eigenvectors of A
    [~,ix] = min(abs(diag(D)-1)); % Locate an eigenvalue which equals 1
    v = V(:,ix)'; % The corresponding row of V' will be a solution
    LUT(x + 1, :) = v/sum(v); % Adjust it to have a sum of 1
end

LUT = round(LUT .* 16);

filename = '8_state_fsm_steady_state_distribution.xlsx';
writematrix(LUT,filename,'Sheet',1)