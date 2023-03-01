function [Z1, Z2, Z2_Test] = project2Domains_v2(filepath, in_filename, out_filename, save_data)
    % PROJECT2DOMAINS Project data of 2 domains into latent space using KEMA RBF kernel
    % Remember that labels start from 1, Label 0 means no label.

    addpath('kema_code/')

	data_path = fullfile(filepath, in_filename);
	load(data_path)

	options.graph.nn = 10;  %KNN graph number of neighbors
	mu = 0.5;               %(1-mu)*L  + mu*(Ls)
	NF = 100; %GT: Max. no. of latent features allowed

	X1 = X1';
	X2 = X2';
    X2_Test = X2_Test';
    
	Y = [Y1;Y2];
	[d1, n1] = size(X1);
	[d2, n2] = size(X2);

	% 2) Compute RBF kernels
	% pdist: Pairwise distance between pairs of observations
	% 1st domain
	kernel_name = 'rbf'; %lin, poly, rbf, sam, chi2
	%% KEMA - KERNEL
	kernel_disp = ['Mapping with the kernel: ', kernel_name];
	disp(kernel_disp)
	sigma1 = 15*mean(pdist(X1'));
	% K1 = kernelmatrix('rbf',[X1],[X1],sigma1);
	K1 = kernelmatrix(kernel_name,[X1],[X1],sigma1);

	% 2nd domain
	sigma2 = 15*mean(pdist(X2'));
	% K2 = kernelmatrix('rbf',[X2],[X2],sigma2);
	K2 = kernelmatrix(kernel_name,[X2],[X2],sigma2);

	% blkdiag: Block diagonal matrix
	K = blkdiag(K1,K2);
    
    % KT2 = kernelmatrix('rbf',[X2],[X2_Test],sigma2);
    KT2 = kernelmatrix(kernel_name,[X2],[X2_Test],sigma2);
    
	%%%%%%% COMPUTE A AND B

	% 2) graph Laplacians
	G1 = buildKNNGraph([X1]',options.graph.nn,1);
	G2 = buildKNNGraph([X2]',options.graph.nn,1);
	W = blkdiag(G1,G2);
	W = double(full(W));
	clear G*

	% Class Graph Laplacian
	Ws = repmat(Y,1,length(Y)) == repmat(Y,1,length(Y))'; Ws(Y == 0,:) = 0; Ws(:,Y == 0) = 0; Ws = double(Ws);
	Wd = repmat(Y,1,length(Y)) ~= repmat(Y,1,length(Y))'; Wd(Y == 0,:) = 0; Wd(:,Y == 0) = 0; Wd = double(Wd);

	Sws = sum(sum(Ws));
	Sw = sum(sum(W));
	Ws = Ws/Sws*Sw;

	Swd = sum(sum(Wd));
	Wd = Wd/Swd*Sw;

	Ds = sum(Ws,2); Ls = diag(Ds) - Ws;
	Dd = sum(Wd,2); Ld = diag(Dd) - Wd;
	D = sum(W,2); L = diag(D) - W;

	% Tune the generalized eigenproblem
	A = ((1-mu)*L  + mu*(Ls)); % (n1+n2) x (n1+n2) %  
	B = Ld;         % (n1+n2) x (n1+n2) %        
	%%%%%%%

	KAK = K*A*K;
	KBK = K*B*K;

	% 3) Extract all features (now we can extract n dimensions!)
	[ALPHA, LAMBDA] = gen_eig(KAK,KBK,'LM');
	[LAMBDA, j] = sort(diag(LAMBDA));
	ALPHA = ALPHA(:,j);

	% 4) Project the data
	nVectRBF = min(NF,rank(KBK));
	nVectRBF = min(nVectRBF,rank(KAK));

	E1 = ALPHA(1:n1, 1:nVectRBF);
	E2 = ALPHA(n1+1:end,1:nVectRBF);

	Phi1toF = E1'*K1;
	Phi2toF = E2'*K2;
    
    Phi2TtoF = E2'*KT2;

	% 5) IMPORTAT: Normalize!!!!
	% m1 = mean(Phi1toF');
	m2 = mean(Phi2toF');
	% s1 = std(Phi1toF');
	s2 = std(Phi2toF');
	
	Phi1toF = zscore(Phi1toF')';
	Phi2toF = zscore(Phi2toF')';
	% OR
	% GT: T = length(XT1)/2; T depends on XT1, but why T is also used for Phi2TtoF and Phi3TtoF ??
	% T = length(XT1)/2;
	% Phi1TtoF = ((Phi1TtoF' - repmat(m1,2*T,1))./ repmat(s1,2*T,1))';
    % Phi2TtoF = ((Phi2TtoF' - repmat(m2,2*T,1))./ repmat(s2,2*T,1))';
    % Phi3TtoF = ((Phi3TtoF' - repmat(m3,2*T,1))./ repmat(s3,2*T,1))';    
    % T = length(X3_Test)/2;
    T = size(X2_Test);
    T = T(2)/2;
    Phi2TtoF = ((Phi2TtoF' - repmat(m2,2*T,1))./ repmat(s2,2*T,1))';

	% GT: Size = no. of examples x no. of features
	Z1 = Phi1toF';
	Z2 = Phi2toF';
    Z2_Test = Phi2TtoF';

	if save_data
		data_path = fullfile(filepath, out_filename);
		save(data_path,'Z1', 'Z2', 'Z2_Test', '-v6')

end
