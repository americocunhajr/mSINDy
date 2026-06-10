% -----------------------------------------------------------------
%  mSINDy.m
% ----------------------------------------------------------------- 
%  Programmer: Americo Cunha Jr
%              americo.cunhajr@gmail.com
%
%  Originally programmed in: Jul 07, 2021
%            Last update in: May 26, 2026
% -----------------------------------------------------------------
%  This class implements the Sparse Identification of Nonlinear 
%  Dynamical Systems (SINDy)  algorithm  to  compute  a  sparse
%  evolution law (vector field) for a certain autonomous dynamical 
%  system in which a (suficiently informative) time-series for the 
%  dynamic state and its derivative/first iterate are known, i.e., 
%  SINDy finds f(x) for a dynamical system like
%  
%         dx/dt = f(x)                 x_{n+1} = f(x_{n})
%       + initial condition    or    + initial condition
%   
%  given the time-series for (x,dxdt) or (x_{n},x_{n+1}).
%
%  Methods:
%   Run                 - Executes the SINDy algorithm to compute 
%                         the regression coefficients
%   Dictionary          - Constructs a library of functions for the
%                         underlying regression process
%   SolverSTLS          - Sequential Threshold Least Squares solver
%   SolverLASSO         - LASSO solver for sparse regression
%   ValidateInput       - Validates the user-specified options
%   ShowWarning         - Displays warnings for high order terms
%   Combinations_k_by_k - Generates combinations
% -----------------------------------------------------------------
% Reference:
%   S. L. Brunton, J. L. Proctor, and J. N. Kutz, "Discovering
%   governing equations from data by sparse identification of
%   nonlinear dynamical systems," Proceedings of the National
%   Academy of Sciences, Vol. 113, pp. 3932-3937, 2016.
% -----------------------------------------------------------------
classdef mSINDy
    
    % .................................................................
    properties
        order_poly  = 3;      % Polynomial    order for dictionary
        order_trig  = 0;      % Trigonometric order for dictionary
        order_exp   = 0;      % Exponential   order for dictionary
        order_log   = 0;      % Logarithmic   order for dictionary
        solver      = 'stls'; % Default solver
        lambda      = 1e-3;   % Default threshold parameter
        Nthresh     = 10;     % Default number of thresholding events
        verbose     = true;   % Show warnings by default
    end
    % .................................................................
    

    methods
        
        % Constructor to initialize SINDy with user-specified options
        % .................................................................
        function SINDy = mSINDy(Opts)

            if nargin > 0
                SINDy = SINDy.ValidateInput(Opts);
            end
        end
        % .................................................................
        
        
        % .................................................................
        % Validates the user-specified input information
        % .................................................................
        function SINDy = ValidateInput(SINDy,Opts)
            
            % Fields names for mSINDy data structure
            fields = {'order_poly', ...
                      'order_trig', ...
                      'order_exp' , ...
                      'order_log' , ...
                      'solver'    , ...
                      'lambda'    , ...
                      'Nthresh'   , ...
                      'verbose'};
            
            % % Fill empty fields in mSINDy data structure
            for i = 1:length(fields)
                if isfield(Opts,fields{i})
                    SINDy.(fields{i}) = Opts.(fields{i});
                end
            end

            % Make solver name lower-case
            SINDy.solver = lower(SINDy.solver);

            % Check solver name validity
            if ~ismember(SINDy.solver,{'stls','lasso'})
                error('solver must be either ''stls'' or ''lasso''.');
            end

            % Check lambda value validity
            if SINDy.lambda < 0
                error('lambda must be non-negative.');
            end

            % Check Nthresh value validity
            if mod(SINDy.Nthresh,1) ~= 0 || SINDy.Nthresh <= 0
                error('Nthresh must be a positive integer.');
            end

            % Validate dictionary order options
            ValidateDictionary(SINDy);
        end
        % .................................................................
        

        % .................................................................
        % Computes the regression coefficients via SINDy algorithm
        % Input:
        %   X    - (Ndata x Nvars) state coordinates time-series
        %   dXdt - (Ndata x Nvars) state coordinates derivative time-series
        % Output:
        %   Coeffs - (Nbasis x Nvars) regression coefficients
        %   Dict   - (Ndata x Nbasis) dictionary evaluated at X
        % .................................................................
        function [Coeffs, Dict] = Run(SINDy,X,dXdt)

            % dimensions of X
            [Ndata, Nvars] = size(X);

            % check argument for error
            if [Ndata, Nvars] ~= size(dXdt)
                error('X and dXdt must have the same dimensions');
            end
            
            % Construct the dictionary of functions
            Dict = SINDy.Dictionary(X);
            
            % Compute the regression coefficients matrix
            switch SINDy.solver
                case 'lasso'
                    Coeffs = SINDy.SolverLASSO(Dict,dXdt);
                case 'stls'
                    Coeffs = SINDy.SolverSTLS(Dict,dXdt);
                otherwise
                    error('Unknown solver.');
            end
        end
        % .................................................................
        

        % .................................................................
        % Constructs the dictionary (library) of functions for regression
        % .................................................................
        function Dict = Dictionary(SINDy,X)

            % Construct dictionaries for different function types
            POLY_X = SINDy.DictionaryPoly(X);
            TRIG_X = SINDy.DictionaryTrig(X);
            EXP_X  = SINDy.DictionaryExp (X);
            LOG_X  = SINDy.DictionaryLog (X);

            % Combine all dictionaries
            Dict = [POLY_X, TRIG_X, EXP_X, LOG_X];
        end
        % .................................................................
        

        % .................................................................
        %  Validates the dictionary of functions prescribed by the user.
        % .................................................................
        function ValidateDictionary(SINDy)

            % Validate polynomial order
            SINDy.ValidateOrder(SINDy.order_poly,'Polynomial');
            
            % Validate trigonometric order
            SINDy.ValidateOrder(SINDy.order_trig,'Trigonometric');
            
            % Validate exponential order
            SINDy.ValidateOrder(SINDy.order_exp,'Exponential');
            
            % Validate logarithmic order
            SINDy.ValidateOrder(SINDy.order_log,'Logarithmic');
        end
        % .................................................................
        

        % .................................................................
        %  Validates the orders of each prescribed dictionary.
        % .................................................................
        function ValidateOrder(~,order,name)
            
            if isinf(order) || mod(order,1) ~= 0 || order < 0
                error('%s order must be a non-negative integer.',name);
            end
        end
        % .................................................................
        

        % .................................................................
        %  Defines a dictionary of polynomial functions.
        % .................................................................
        function POLY_X = DictionaryPoly(SINDy,X)
        
            % check for very high orders
            SINDy.ShowWarning(SINDy.order_poly,5,'Polynomial',SINDy.verbose);
        
            % data matrix dimensions
            [Ndata,Nvars] = size(X);
        
            % Initialize the dictionary with constant term (order 0)
            POLY_X = ones(Ndata,1);
            
            % Construct polynomial terms up to the specified order
            for k = 1:SINDy.order_poly
                Nterms    = nchoosek(Nvars+k-1,k);
                XPk       = zeros(Ndata,Nterms);
                ColsCombs = SINDy.Combinations_k_by_k(1:Nvars,k);
                for j = 1:Nterms
                    XPk(:,j)  = prod(X(:,ColsCombs(j,:)),2);
                end
                POLY_X = [POLY_X, XPk];
            end
        end
        % .................................................................
        

        % .................................................................
        %  Defines a dictionary of trigonometic functions.
        % .................................................................
        function TRIG_X = DictionaryTrig(SINDy,X)
        
            % check for very high orders
            SINDy.ShowWarning(SINDy.order_trig,5,'Trigonometric',SINDy.verbose);
        
            % define trigonometric dictionary
            TRIG_X = [];
            if SINDy.order_trig > 0
                for k = 1:SINDy.order_trig
                    TRIG_X = [TRIG_X cos(k*X) sin(k*X)];
                end
            end
        end
        % .................................................................
        

        % .................................................................
        %  Defines a dictionary of exponential functions.
        % .................................................................
        function EXP_X = DictionaryExp(SINDy,X)
        
            % check for very high orders
            SINDy.ShowWarning(SINDy.order_exp,2,'Exponential',SINDy.verbose);
        
            % define exponential dictionary
            EXP_X = [];
            
            % Overflow limit for an exponential
            EXP_LIMIT = log(realmax('double'));

            if SINDy.order_exp > 0
                for k = 1:SINDy.order_exp
                    
                    maxArg = max(abs(k*X),[],'all');

                    if SINDy.verbose && maxArg > 0.95*EXP_LIMIT
                        warning(['exp(%d*X) argument reaches %.2f. ' ...
                                 'Overflow may occur (double limit ≈ %.2f).'], ...
                                 k,maxArg,EXP_LIMIT);
                    end

                    EXP_X = [EXP_X exp(k*X)];
                end
            end
        end
        % .................................................................
        

        % .................................................................
        %  Defines a dictionary of logarithmic functions.
        % .................................................................
        function LOG_X = DictionaryLog(SINDy,X)
        
            % check for very high orders
            SINDy.ShowWarning(SINDy.order_log,5,'Logarithmic',SINDy.verbose);
        
            % define logarithmic dictionary
            LOG_X = [];
            if SINDy.order_log > 0
                for k = 1:SINDy.order_log
                    LOG_X = [LOG_X log(abs(k*X)+eps)];
                end
            end
        end
        % .................................................................


        % .................................................................
        % Displays a warning message for high orders if enabled
        % .................................................................
        function ShowWarning(~,order,threshold,dictType,verbose)
            
            if verbose && order > threshold
                warning(['\nThe %s order is greater than %d. ' ...
                          'This may cause memory issues or ' ...
                          'long computation times.\n'],dictType,threshold);
                % fprintf(['\nThe %s order is greater than %d. ' ...
                %          'This may cause memory issues or ' ...
                %          'long computation times.\n'],dictType,threshold);
            end
        end
        % .................................................................
        

        % .................................................................
        % Generates all possible sorted k-combinations with repetition
        % .................................................................
        function combs = Combinations_k_by_k(SINDy,indices,k)

            if k == 0
                combs = [];
                return;
            end

            if isempty(indices)
                combs = [];
            else
                combs = [];
                for i = 1:length(indices)
                    if k > 1
                        subcombs = SINDy.Combinations_k_by_k(indices(i:end),k-1);
                        combs    = [combs; ...
                                   [indices(i)*ones(size(subcombs,1),1),subcombs]];
                    else
                        combs = [combs; indices(i)];
                    end
                end
            end
        end
        % .................................................................

        % .................................................................
        %  Normalizes the columns of a matrix using the Euclidean norm.
        % .................................................................
        function [A_norm,ColsNorms] = MyNormC(~,A)
            
            % Check for input errors
            if ~ismatrix(A)
                error('The input must be a matrix.')
            end

            if ~isnumeric(A)
                error('The input must be numeric.')
            end
            
            % compute Euclidean norm of each column
            ColsNorms = sqrt(sum(A.^2,1));
            
            % avoid division by zero
            ColsNormsSafe                       = ColsNorms;
            ColsNormsSafe(ColsNormsSafe == 0.0) = 1.0;
            
            % normalize columns
            A_norm = A./ColsNormsSafe;
        end
        % .................................................................

        % .................................................................
        % Sequential Threshold Least Squares solver
        % .................................................................
        function Coeffs = SolverSTLS(SINDy,A,B)
            
            [m,p]  = size(A);
            n      = size(B,2);
            
            if m ~= size(B,1)
                error('A and B must have the same number of rows.');
            end
            
            [A_norm,ColsNorms] = SINDy.MyNormC(A);
            
            X = A_norm\B;
            
            n_thresh = 0;

            while n_thresh < SINDy.Nthresh
                
                small_coeff = abs(X) < SINDy.lambda;
                
                X(small_coeff) = 0.0;

                for j = 1:n
                    
                    big_coeff = ~small_coeff(:,j);
                    
                    if any(big_coeff)
                        X(big_coeff,j) = A_norm(:,big_coeff)\B(:,j);
                    end
                end

                n_thresh = n_thresh + 1;
            end

            for k = 1:size(X,1)
                if ColsNorms(k) > 0.0
                    X(k,:) = X(k,:)/ColsNorms(k);
                else
                    X(k,:) = 0.0;
                end
            end
            
            Coeffs = X;
        end
        % .................................................................
        

        % .................................................................
        % Runs LASSO to find the sparse coefficients
        % .................................................................
        function Coeffs = SolverLASSO(SINDy,A,B)
            [m,p]  = size(A);
            n      = size(B,2);

            if m ~= size(B,1)
                error('A and B must have the same number of rows.');
            end

            Coeffs = zeros(p,n);
            for k = 1:n
                Coeffs(:,k) = SINDy.MyLasso(A,B(:,k),SINDy.lambda);
                % [Coeffs_aux,FitInfo] = lasso(A,B(:,k),'CV',SINDy.Nthresh,...
                %                                       'Intercept',false);
                %          Coeffs(:,k) = Coeffs_aux(:,FitInfo.IndexMinMSE);
            end
        end
        % .................................................................

        % .................................................................
        %  MyLasso.m
        % .................................................................
        %  Solves the LASSO problem
        %
        %      min_x  1/2 ||b - A*x||_2^2 + lambda ||x||_1
        %
        %  using coordinate descent.
        %
        %  Input:
        %   A        - (m x p) regression matrix
        %   b        - (m x 1) target vector
        %   lambda   - non-negative regularization parameter
        %   maxIter  - maximum number of iterations, optional
        %   tol      - convergence tolerance, optional
        %
        %  Output:
        %   x        - (p x 1) sparse coefficient vector
        % .................................................................
        function x = MyLasso(SINDy,A,b,lambda,maxIter,tol)
        
            if nargin < 4
                error('Too few inputs.')
            elseif nargin > 6
                error('Too many inputs.')
            end
        
            if nargin < 5 || isempty(maxIter)
                maxIter = 1000;
            end
        
            if nargin < 6 || isempty(tol)
                tol = 1e-8;
            end
        
            [m,p] = size(A);
        
            if size(b,1) ~= m || size(b,2) ~= 1
                error('b must be a column vector with the same number of rows as A.')
            end
        
            if lambda < 0
                error('lambda must be non-negative.')
            end
        
            if mod(maxIter,1) ~= 0 || maxIter <= 0
                error('maxIter must be a positive integer.')
            end
        
            if tol <= 0
                error('tol must be positive.')
            end
        
            % Normalize columns
            [A_norm,colNorms] = SINDy.MyNormC(A);
        
            % Initialize coefficients
            x_norm = zeros(p,1);
        
            % Precompute squared column norms
            colSqNorms = sum(A_norm.^2,1);
        
            % Residual
            r = b - A_norm*x_norm;
        
            for iter = 1:maxIter
        
                x_old = x_norm;
        
                for j = 1:p
        
                    if colSqNorms(j) == 0
                        x_norm(j) = 0;
                        continue
                    end
        
                    % Add back contribution of current coefficient
                    r = r + A_norm(:,j)*x_norm(j);
        
                    % Coordinate-wise least-squares correlation
                    rho = A_norm(:,j)'*r;
        
                    % Soft-thresholding update
                    x_norm(j) = SINDy.SoftThreshold(rho,lambda)/colSqNorms(j);
        
                    % Update residual
                    r = r - A_norm(:,j)*x_norm(j);
                end
        
                if norm(x_norm - x_old,inf) < tol
                    break
                end
            end
        
            % Return coefficients in original scale
            colNormsSafe = colNorms;
            colNormsSafe(colNormsSafe == 0) = 1;
        
            x = x_norm ./ colNormsSafe';
        
            x(colNorms == 0) = 0;
        end
        % .................................................................
        
        % .................................................................
        %  Soft-thresholding operator
        % .................................................................
        function y = SoftThreshold(~,z,gamma)
        
            if z > gamma
                y = z - gamma;
            elseif z < -gamma
                y = z + gamma;
            else
                y = 0.0;
            end
        end
        % .................................................................

        % .................................................................
        % Evolution law for the data-driven model
        % .................................................................
        function F = EvolutionLaw(SINDy,Coeffs,~,s)
            s = s(:).';
            F = (SINDy.Dictionary(s)*Coeffs).';
        end
        % .................................................................

        % .................................................................
        function CoeffList = GenerateCoeffList(SINDy,Coeffs,VarNames)
        
            % check number of arguments
            if nargin < 3
                error('Too few inputs.');
            elseif nargin > 3
                error('Too many inputs.');
            end
        
            % convert to column format (if necessary)
            VarNames = VarNames(:);
            
            % regression coefficients matrix dimensions
            [Ndict,Nvars] = size(Coeffs);
            
            % check for consistency
            if size(VarNames,1) ~= Nvars
                error('VarNames list must have Nvars entries.');
            end
        
            % preallocate memory for BasisList cell struct
            BasisList = cell(Ndict,1);
        
            % Initialize the index
            ii = 1;
        
            % Generate polynomial basis functions
            if SINDy.order_poly >= 0
                [BasisList,ii] = SINDy.BasisListPoly(BasisList,VarNames,SINDy.order_poly,ii);
                %ii = ii + length(BasisList);
            end
        
            % Generate trigonometric basis functions
            if SINDy.order_trig > 0
                [BasisList,ii] = SINDy.BasisListTrig(BasisList,VarNames,SINDy.order_trig,ii);
                %ii = ii + length(BasisList);
            end
        
            % Generate exponential basis functions
            if SINDy.order_exp > 0
                [BasisList,ii] = SINDy.BasisListExp(BasisList,VarNames,SINDy.order_exp,ii);
            end
        
            % Generate logarithmic basis functions
            if SINDy.order_log > 0
                [BasisList,ii] = SINDy.BasisListLog(BasisList,VarNames,SINDy.order_log,ii);
            end
            
            if ii-1 ~= Ndict
                error('Number of generated basis labels does not match dictionary size.');
            end

            % preallocate memory for CoeffList cell struct
            CoeffList = cell(Ndict+1,Nvars+1);
        
            % first line of CoeffList struct
            CoeffList{1,1} = 'Basis';
            for i=1:Nvars
                CoeffList{1,i+1} = ['d',VarNames{i},'dt'];
            end
        
            % format to show the numerical coefficients
            formatSpec = '%+.6E';
        
            % other entries of CoeffList cell struct
            for i=1:Ndict

                % first column of CoeffList cell struct
                CoeffList{i+1,1} = BasisList{i};

                for j=1:Nvars

                    % other columns of CoeffList cell struct
                    CoeffList{i+1,j+1} = num2str(Coeffs(i,j),formatSpec);
                end
            end
        end
        % .................................................................
        

        % .................................................................
        %  Generates polynomial basis functions up to a specified order.
        % .................................................................
        function [BasisList,ii] = BasisListPoly(SINDy,BasisList,VarNames,maxOrder,ii)

            % Order 0 polynomial
            if maxOrder >= 0

                % Constant term
                BasisList{ii} = '1'; 
                ii = ii + 1;
            end

            % Number of variables
            Nvars = length(VarNames);

            % Generate higher order polynomial terms
            for order = 1:maxOrder
                
                % Combinations of polynomials terms
                combs = SINDy.Combinations_k_by_k(1:length(VarNames),order);

                for j = 1:size(combs,1)
        
                    powers = histcounts(combs(j,:),1:(Nvars+1));
        
                    pieces = {};
        
                    for v = 1:Nvars
                        if powers(v) == 1

                            pieces{end+1} = VarNames{v};
                        elseif powers(v) > 1

                            pieces{end+1} = sprintf('%s^%d',VarNames{v},powers(v));
                        end
                    end

                    BasisList{ii} = strjoin(pieces,'*');
                    ii = ii + 1;
                end
            end
        end
        % .................................................................
        
        % .................................................................
        %  Generates trigonometric basis functions up to a specified order.
        % .................................................................
        function [BasisList,ii] = BasisListTrig(~,BasisList,VarNames,maxOrder,ii)
            for order = 1:maxOrder
                for i = 1:length(VarNames)
                    BasisList{ii} = ['cos(',num2str(order),'*',VarNames{i},')'];
                    ii = ii + 1;
                end
                for i = 1:length(VarNames)
                    BasisList{ii} = ['sin(',num2str(order),'*',VarNames{i},')'];
                    ii = ii + 1;
                end
            end
        end
        % .................................................................
        
        % .................................................................
        %  Generates exponential basis functions up to a specified order.
        % .................................................................
        function [BasisList,ii] = BasisListExp(~,BasisList,VarNames,maxOrder,ii)
            for order = 1:maxOrder
                for i=1:length(VarNames)
                    BasisList{ii} = ['exp(',num2str(order),VarNames{i},')'];
                    ii = ii + 1;
                end
            end
        end
        % .................................................................
        
        % .................................................................
        %  Generates logarithmic basis functions up to a specified order.
        % .................................................................
        function [BasisList,ii] = BasisListLog(~,BasisList,VarNames,maxOrder,ii)
            for order = 1:maxOrder
                for i=1:length(VarNames)
                    BasisList{ii} = ['log(abs(',num2str(order),VarNames{i},'))'];
                    ii = ii + 1;
                end
            end
        end
        % .................................................................
    end
end
% -------------------------------------------------------------------------
