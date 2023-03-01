function [U D,n_eig] = gen_eig(A,B,option,n_eig)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extracts generalized eigenvalues for problem A * U = B * U * Landa
% n_eig -- Number of eigenvalues to compute (optional)
% option = 'LM' or 'SM'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (nargin==4)
    n_eig = min([n_eig rank(A) rank(B)]);
else
    n_eig = min([rank(A) rank(B)]);
end
OPTS.disp = 0;

B = (B + B')/2;
R = size(B,1);
rango = rank(B);
if (rango == R)
    U = zeros(R,n_eig);
    D = zeros(n_eig,n_eig);
    inv_B = inv(B);
    for k = 1:n_eig
        [a,d] = eigs(inv_B*A,1,option,OPTS);
        a = a ./ sqrt(a'*B*a);
        U(:,k) = a;
        D(k,k) = d;
        A = A - d * (B * a) * (a' * B);
    end
else
%     rango = max(rango,sum(eig(B)>0) - 5);
    B = double(B); % GT: First argument must be a double matrix or a function
    [v,d] = eigs(B,rango);
    B = v'*B*v;
    A = v'*A*v;
    U2 = zeros(rango,n_eig);
    D = zeros(n_eig,n_eig);
    inv_B = inv(B);
    for k = 1:n_eig
        try
            A = double(A); % GT: First argument must be a double matrix or a function

            % GT: reset the lastwarn message and id
            lastwarn('', '');

            [a,d] = eigs(inv_B*A,1,option,OPTS);

            % GT: now if a warning was raised, warnMsg and warnId will not be empty.
            [warnMsg, warnId] = lastwarn();

            % GT:
            if ~(isempty(warnId))
                fprintf('There was an warning: %s\n', warnMsg);
                fprintf('The warnId was: %s\n', warnId);
                switch warnId
                    case 'MATLAB:eigs:NotAllEigsConverged'
                        % GT: Warning: 0 of the 1 requested eigenvalues converged. Eigenvalues that did not converge are NaN.
                        % Solution: https://www.mathworks.com/matlabcentral/answers/172633-eig-doesn-t-converge-can-you-explain-why

                        A_gt = inv_B*A;
                        % First we compute the squared Frobenius norm of our matrix
                        nA = sum(sum(A_gt.^2));
                        % Then we make this norm be meaningful for element wise comparison
                        nA = nA / numel(A_gt);
                        % Finally, we smooth our matrix
                        As = A_gt;
                        As( As.^2 < 1e-10*nA ) = 0;

                        [a,d] = eigs(As,1,option,OPTS);
                end
            end
        catch err
            fprintf('There was an error: %s\n', err.message);
            fprintf('The identifier was: %s\n', err.identifier);

            n_eig = k;
            return
        end
        a = a ./ sqrt(a'*B*a);
        U2(:,k) = a;
        D(k,k) = d;
        A = A - d * (B * a) * (a' * B);
    end
    U = v * U2;
end
