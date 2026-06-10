% -------------------------------------------------------------------------
%  Example03_VanDerPolOscillator.m
% -------------------------------------------------------------------------
%  Programmer: Americo Cunha Jr
%              americo@lncc.br
%
%  Originally programmed in: May 16, 2026
%            Last update in: May 29, 2026
% -------------------------------------------------------------------------
%  Sparse Identification of Nonlinear Dynamics (SINDy)
%  Example 3: van der Pol oscillator
%
%  This example demonstrates sparse equation discovery for the classical
%  van der Pol oscillator using the mSINDy framework.
%
%  The van der Pol system constitutes an important nonlinear benchmark for
%  sparse identification because the governing equations contain nonlinear
%  state-dependent damping is responsible for self-excited oscillatory
%  behavior and stable limit-cycle dynamics.
%
%  Governing equations:
%
%       dxdt = y
%
%       dydt = mu*(1 - x^2)*y - x
%
%  where:
%       x  -> displacement-like state variable
%       y  -> velocity-like state variable
%       mu -> nonlinear damping parameter
%
%  Required file in MATLAB path:
%    mSINDy.m
% -------------------------------------------------------------------------

clc
clear
close all


% -------------------------------------------------------------------------
% Program header
% -------------------------------------------------------------------------
disp(' ')
disp('===============================================================')
disp(' mSINDy                                                        ')
disp(' Sparse Identification of Nonlinear Dynamics                   ')
disp(' by                                                            ')
disp(' Americo Cunha Jr                                              ')
disp('                                                               ')
disp(' Example 3: van der Pol oscillator                             ')
disp('===============================================================')
disp(' ')

case_name = 'Example03_VanDerPolOscillator';
fig_dir   = 'figures';

if ~exist(fig_dir,'dir')
    mkdir(fig_dir);
end
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Fix the seed for reproducibility
% -------------------------------------------------------------------------
rng_stream = RandStream('mt19937ar','Seed',30081984);
RandStream.setGlobalStream(rng_stream);
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% System parameters
% -------------------------------------------------------------------------
Param.mu = 1.5;    % nonlinear damping parameter

% initial conditions
IC_old = [2.0; 0.0];
IC_new = [-1.5; 2.0];


% Time interval
t0 = 0.0;
t1 = 80.0;

% Number of temporal samples
Ntime = 4001;

% Uniform time grid
time = linspace(t0,t1,Ntime).';

% dynamical system evolution law
TrueEvolutionLaw = @(t,s) dsdt_VDP(t,s,Param);
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Training data
% -------------------------------------------------------------------------
t0_train   = t0;
t1_train   = 0.1*t1;
dt_train   = time(2) - time(1);
time_train = (t0_train:dt_train:t1_train)';

% Integrate reference dynamics
[~,X_train] = ode45(TrueEvolutionLaw,time_train,IC_old);


% Measurement noise intensity
noise_level = 0.01;

% Additive Gaussian measurement noise
X_train = X_train + noise_level*std(X_train).*randn(size(X_train));
% -------------------------------------------------------------------------


% Compute derivative data
% -------------------------------------------------------------------------
disp(' ')
disp(' --- computing derivative data --- ')
disp(' ')

% Exact derivatives evaluated from the governing equations
N          = size(X_train,1);
dXdt_train = zeros(size(X_train));

for n = 1:N
    dXdt_train(n,:) = dsdt_VDP(0.0,X_train(n,:),Param).';
end

% Add derivative noise
 dXdt_train = dXdt_train + noise_level*randn(size(dXdt_train));
% -------------------------------------------------------------------------


% Configure mSINDy options
% -------------------------------------------------------------------------
disp(' ')
disp(' --- configuring sparse identification --- ')
disp(' ')

Opts.lambda     = 0.01;
Opts.Nthresh    = 15;
Opts.solver     = 'stls';
Opts.order_poly = 3;
Opts.order_trig = 0;
Opts.order_exp  = 0;
Opts.order_log  = 0;
Opts.verbose    = true;

MySINDyModel = mSINDy(Opts);
% -------------------------------------------------------------------------


% Sparse identification
% -------------------------------------------------------------------------
disp(' ')
disp(' --- sparse equation discovery --- ')
disp(' ')

[XI,THETA] = MySINDyModel.Run(X_train,dXdt_train);

CoeffList = MySINDyModel.GenerateCoeffList(XI,{'x','y'});

fprintf('Identified sparse coefficients:\n')
disp(CoeffList)

% Data-driven identified model
DataDrivenModel = @(t,s) MySINDyModel.EvolutionLaw(XI,t,s);
% -------------------------------------------------------------------------


% Validation on training trajectory
% -------------------------------------------------------------------------
disp(' ')
disp(' --- validating on training trajectory --- ')
disp(' ')

[~,X_true]  = ode45(TrueEvolutionLaw,time,IC_old);
[~,X_model] = ode45(DataDrivenModel ,time,IC_old);
% -------------------------------------------------------------------------


% Validation on unseen initial condition
% -------------------------------------------------------------------------
disp(' ')
disp(' --- validating on unseen initial condition --- ')
disp(' ')

[~,X_true_new] = ode45(TrueEvolutionLaw,time,IC_new);
[~,X_model_new] = ode45(DataDrivenModel,time,IC_new);
% -------------------------------------------------------------------------


% Relative validation errors
% -------------------------------------------------------------------------
err_train = RelativeError(X_true,X_model);
err_test  = RelativeError(X_true_new,X_model_new);

fprintf('Relative trajectory error (training IC): %10.4e',err_train)
fprintf(' Relative trajectory error (unseen IC) : %10.4e',err_test)
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Figure formatting parameters
% -------------------------------------------------------------------------
set(groot,'defaultTextInterpreter'         ,'latex')
set(groot,'defaultAxesTickLabelInterpreter','latex')
set(groot,'defaultLegendInterpreter'       ,'latex')
set(groot,'defaultAxesFontName'            ,'Times')
set(groot,'defaultTextFontName'            ,'Times')
set(groot,'defaultAxesFontSize'            ,18     )
set(groot,'defaultLineLineWidth'           ,2.5    )

% Color palette, consistent with Duffing example
clr_true  = [0.85 0.00 0.00];  % red
clr_model = [0.00 0.00 0.00];  % black
clr_data  = [0.00 0.50 0.00];  % green
clr_phase = [0.00 0.65 0.90];  % cyan
clr_train = [0.00 0.20 0.90];  % blue
clr_test  = [0.90 0.10 0.00];  % red-orange

offset1 = 0.98;
offset2 = 1.02;
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Figure 1 - Oscillator state time series
% -------------------------------------------------------------------------
fig1 = figure('Color','w',...
              'Units','inches',...
              'Position',[1 1 10 5]);

h1 = plot(time,X_true(:,1),'-','Color',clr_true,'LineWidth',2.0);
hold on
h2 = plot(time,offset2*X_model(:,1),'--','Color',clr_model,'LineWidth',1.5);
h3 = plot(time_train,X_train(:,1),'.','Color',clr_data,'MarkerSize',18);
hold off

box on; grid off
xlabel('time')
ylabel('state')
legend([h1 h2 h3],{'Original dynamics~',...
                   'Data-driven model~',...
                   'Training data~'},...
                   'Location','Best')
xlim([t0 t1])
ylim(1.15*[min(X_true(:,1)) max(X_true(:,1))])
set(gca,'LineWidth',1.2,'TickDir','in','XMinorTick','on','YMinorTick','on')

exportgraphics(fig1,fullfile(fig_dir,[case_name,'_disp.pdf']),'ContentType','vector')
exportgraphics(fig1,fullfile(fig_dir,[case_name,'_disp.png']),'Resolution',600)
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Figure 2 - State rate time series
% -------------------------------------------------------------------------
fig2 = figure('Color','w',...
              'Units','inches',...
              'Position',[1 1 10 5]);

h1 = plot(time,X_true(:,2),'-','Color',clr_true,'LineWidth',2.0);
hold on
h2 = plot(time,offset2*X_model(:,2),'--','Color',clr_model,'LineWidth',1.5);
h3 = plot(time_train,X_train(:,2),'.','Color',clr_data,'MarkerSize',18);
hold off

box on; grid off
xlabel('time')
ylabel('state-derivative')
legend([h1 h2 h3],{'Original dynamics~',...
                   'Data-driven model~',...
                   'Training data~'},...
                   'Location','Best')
xlim([t0 t1])
ylim(1.15*[min(X_true(:,2)) max(X_true(:,2))])
set(gca,'LineWidth',1.2,'TickDir','in','XMinorTick','on','YMinorTick','on')

exportgraphics(fig2,fullfile(fig_dir,[case_name,'_velo.pdf']),'ContentType','vector')
exportgraphics(fig2,fullfile(fig_dir,[case_name,'_velo.png']),'Resolution',600)
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Figure 3 - Phase-space reconstruction
% -------------------------------------------------------------------------
fig3 = figure('Color','w',...
              'Units','inches',...
              'Position',[1 1 7 6]);

h1 = plot(X_true(:,1),X_true(:,2),'-','Color',clr_phase,'LineWidth',2.0);
hold on
h2 = plotDashedCurve(offset1*X_model(:,1),offset1*X_model(:,2),'--',...
                     'Color',clr_model,'LineWidth',1.5);
h3 = plot(X_train(:,1),X_train(:,2),'.','Color',clr_data,'MarkerSize',10);
hold off

box on; grid off
xlabel('state')
ylabel('state-derivative')
legend([h1 h2 h3],{'Original dynamics~',...
                   'Data-driven model~',...
                   'Training data~'},...
                   'Location','Best')
axis equal
xlim(1.10*[min(X_true(:,1)) max(X_true(:,1))])
ylim(1.10*[min(X_true(:,2)) max(X_true(:,2))])
set(gca,'LineWidth',1.2,'TickDir','in','XMinorTick','on','YMinorTick','on')

exportgraphics(fig3,fullfile(fig_dir,[case_name,'_phase_space.pdf']),'ContentType','vector')
exportgraphics(fig3,fullfile(fig_dir,[case_name,'_phase_space.png']),'Resolution',600)
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Figure 4 - Validation with unseen initial condition
% -------------------------------------------------------------------------
fig4 = figure('Color','w',...
              'Units','inches',...
              'Position',[1 1 7 6]);

h1 = plot(X_true_new(:,1),X_true_new(:,2),'-','Color',clr_phase,'LineWidth',2.0);
hold on
h2 = plotDashedCurve(offset1*X_model_new(:,1),offset1*X_model_new(:,2),'--',...
                     'Color',clr_model,'LineWidth',1.5);
hold off

box on; grid off
xlabel('state')
ylabel('state-derivative')
legend([h1 h2],{'Novel IC~',...
                'Data-driven model~'},...
                'Location','Best')
axis equal
xlim(1.10*[min(X_true_new(:,1)) max(X_true_new(:,1))])
ylim(1.10*[min(X_true_new(:,2)) max(X_true_new(:,2))])
set(gca,'LineWidth',1.2,'TickDir','in','XMinorTick','on','YMinorTick','on')

exportgraphics(fig4,fullfile(fig_dir,[case_name,'_validation.pdf']),'ContentType','vector')
exportgraphics(fig4,fullfile(fig_dir,[case_name,'_validation.png']),'Resolution',600)
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Figure 5 - Sparse coefficient matrix with active rows only
% -------------------------------------------------------------------------
active_tol  = 1.0e-8;
active_rows = any(abs(XI) > active_tol,2);
XI_active   = XI(active_rows,:);

basis_table   = MySINDyModel.GenerateCoeffList(XI,{'x','y'});
active_labels = basis_table([false; active_rows],1);

fig5 = figure('Color','w',...
              'Units','inches',...
              'Position',[1 1 7.5 4.8]);

imagesc(XI_active)
colormap(redbluecmap(256))
colorbar

cmax = max(abs(XI_active(:)));
if cmax == 0
    cmax = 1;
end
clim([-cmax cmax])

xticks(1:2)
xticklabels({'$\dot{x}$','$\dot{y}$'})
yticks(1:size(XI_active,1))
yticklabels(active_labels)

title('Active sparse coefficient matrix')
set(gca,'LineWidth',1.1,'TickDir','out','FontSize',14)

exportgraphics(fig5,fullfile(fig_dir,[case_name,'_coefficients.pdf']),'ContentType','vector')
exportgraphics(fig5,fullfile(fig_dir,[case_name,'_coefficients.png']),'Resolution',600)
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
fprintf('\nFigures written to folder: %s\n',fig_dir);
fprintf('Example 3 completed successfully.\n\n');
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Simple blue-white-red colormap for coefficient matrices
% -------------------------------------------------------------------------
function cmap = redbluecmap(m)
if nargin < 1
    m = 256;
end

bottom = [0.05 0.20 0.75];
middle = [1.00 1.00 1.00];
top    = [0.80 0.05 0.05];

x = linspace(0,1,m).';
cmap = zeros(m,3);

for i = 1:m
    if x(i) < 0.5
        a = x(i)/0.5;
        cmap(i,:) = (1-a)*bottom + a*middle;
    else
        a = (x(i)-0.5)/0.5;
        cmap(i,:) = (1-a)*middle + a*top;
    end
end
end
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Function to plot a dashed curve in phase space
% -------------------------------------------------------------------------
function h = plotDashedCurve(x,y,varargin)

dash_len = 25;
gap_len  = 14;

n = numel(x);
draw_mask = false(n,1);

k = 1;
draw = true;

while k <= n
    if draw
        idx = k:min(k+dash_len-1,n);
        draw_mask(idx) = true;
        k = k + dash_len;
    else
        k = k + gap_len;
    end
    draw = ~draw;
end

x_plot = x;
y_plot = y;

x_plot(~draw_mask) = NaN;
y_plot(~draw_mask) = NaN;

h = plot(x_plot,y_plot,varargin{:});
end
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Dynamical system evolution law
% -------------------------------------------------------------------------
% State definition:
%  x  = displacement-like state variable
%  y  =     velocity-like state variable
% -------------------------------------------------------------------------
% Governing equations:
% dxdt = y
% dydt = mu*(1 - x^2)*y - x
% -------------------------------------------------------------------------
function dsdt = dsdt_VDP(~,state,param)

    x = state(1);
    y = state(2);

    mu = param.mu;

    dxdt = y;
    dydt = mu*(1 - x^2)*y - x;

    dsdt = [dxdt; dydt];
end
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Relative Frobenius trajectory error
% -------------------------------------------------------------------------
function err = RelativeError(Xref,Xmodel)
    err = norm(Xref - Xmodel,'fro')/norm(Xref,'fro');
end
% -------------------------------------------------------------------------
