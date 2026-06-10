% -------------------------------------------------------------------------
%  Example01_HarmonicOscillator.m
% -----------------------------------------------------------------
%  Programmer: Americo Cunha Jr
%              americo@lncc.br
%
%  Originally programmed in: May 16, 2026
%            Last update in: May 22, 2026
% -----------------------------------------------------------------
%  Sparse Identification of Nonlinear Dynamics (SINDy)
%  Example 1: Linear harmonic oscillator
%
%  This script uses the mSINDy class to identify the governing equations
%  of the linear harmonic oscillator from sampled trajectory data.
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
disp(' Example 1: Linear harmonic oscillator                         ')
disp('===============================================================')
disp(' ')

case_name = 'Example01_HarmonicOscillator';
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
t0            = 0.0;           % initial time
t1            = 60.0;          % final time
dt            = 0.2;           % time step
Param.omega_n = 1.25;          % natural frequency
Param.x0      = 2.5;           % initial displacement
Param.v0      = -1.0;          % initial velocity

% initial conditions
IC = [Param.x0; Param.v0];

% dynamical system evolution law
TrueEvolutionLaw = @(t,s) dsdt_HarmonicOscillator(t,s,Param);
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Training data
% -------------------------------------------------------------------------
t0_train    = t0;
t1_train    = 0.2*t1;
dt_train    = 2*dt;
time_train  = (t0_train:dt_train:t1_train)';

% Integrate true dynamics on the training interval
[~,X_train] = ode45(TrueEvolutionLaw,time_train,IC);

% Dimensions of the dataset
[Ndata,Nvars] = size(X_train);

% Apply noise
sigma = 0.01;
X_train = X_train + sigma*std(X_train).*randn(Ndata,Nvars);

% Compute time series derivatives (from the exact vector field)
dXdt_train = zeros(Ndata,Nvars);
for k = 1:Ndata
    dXdt_train(k,:) = TrueEvolutionLaw(0.0,X_train(k,:)').';
end
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Sparse identification with mSINDy
% -------------------------------------------------------------------------
Opts.lambda     = 1.0e-3;
Opts.Nthresh    = 10;
Opts.solver     = 'stls';
Opts.order_poly = 3;
Opts.order_trig = 0;
Opts.order_exp  = 0;
Opts.order_log  = 0;
Opts.verbose    = true;

MySINDyModel = mSINDy(Opts);

[XI,THETA] = MySINDyModel.Run(X_train,dXdt_train);
CoeffList  = MySINDyModel.GenerateCoeffList(XI,{'x1','x2'});

% Display identified coefficient table
disp('Identified coefficient table:')
disp(CoeffList)

% Data-driven evolution law
DataDrivenModel = @(t,s) MySINDyModel.EvolutionLaw(XI,t,s);
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Validation Test 1: train/test split on the same trajectory
% -------------------------------------------------------------------------
t0_val   = t0;
t1_val   = t1;
dt_val   = dt;
time_val = (t0_val:dt_val:t1_val)';

[~,X_true]  = ode45(TrueEvolutionLaw,time_val,IC);
[~,X_model] = ode45(DataDrivenModel ,time_val,IC);

% Training data shown only over the training interval
train_mask = time_val <= t1_train;
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Validation Test 2: unseen initial conditions
% -------------------------------------------------------------------------
IC_test = [ 1.5, -2.0,  0.5;
            0.0,  1.5, -2.5];

Ntest      = size(IC_test,2);
X_true_ic  = cell(Ntest,1);
X_model_ic = cell(Ntest,1);

for j = 1:Ntest
    [~,X_true_ic{j} ] = ode45(TrueEvolutionLaw,time_val,IC_test(:,j));
    [~,X_model_ic{j}] = ode45(DataDrivenModel, time_val,IC_test(:,j));
end
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Compute error metrics
% -------------------------------------------------------------------------
rel_err_train = norm(X_true(train_mask,:) - X_model(train_mask,:),'fro') / ...
                norm(X_true(train_mask,:),'fro');
rel_err_test  = norm(X_true(~train_mask,:) - X_model(~train_mask,:),'fro') / ...
                norm(X_true(~train_mask,:),'fro');

fprintf('\nRelative error on training interval : %.3e\n',rel_err_train);
fprintf('Relative error on test interval     : %.3e\n'  ,rel_err_test );
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Figure formating parameters
% -------------------------------------------------------------------------
set(groot,'defaultTextInterpreter'         ,'latex')
set(groot,'defaultAxesTickLabelInterpreter','latex')
set(groot,'defaultLegendInterpreter'       ,'latex')
set(groot,'defaultAxesFontName'            ,'Times')
set(groot,'defaultTextFontName'            ,'Times')
set(groot,'defaultAxesFontSize'            ,18     )
set(groot,'defaultLineLineWidth'           ,2.5    )

% Color palette
clr_true  = [0.85 0.00 0.00];  % red
clr_model = [0.00 0.00 0.00];  % black
clr_data  = [0.00 0.50 0.00];  % green
clr_phase = [0.00 0.65 0.90];  % cyan

% Small visual offset for data-driven curves
offset = 0.98;

% Number of points corresponding to one oscillation period
T_period = 2*pi/Param.omega_n;
Nperiod  = find(time_val <= time_val(1) + T_period,1,'last');
idx_period = 1:Nperiod;
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Figure 1: displacement validation
% -------------------------------------------------------------------------
fig1 = figure('Color','w',...
             'Units','inches',...
             'Position',[1 1 10 5]);
h1 =  plot(time_val  ,X_true(:,1) ,'-' ,'Color',clr_true ,'LineWidth' ,2.0);
hold on
h2 =  plot(time_val  ,X_model(:,1),'--','Color',clr_model,'LineWidth' ,1.5);
h3 =  plot(time_train,X_train(:,1),'.' ,'Color',clr_data ,'MarkerSize',22);
h4 = xline(t1_train  ,'--','Color',[0.35 0.35 0.35],...
                      'LineWidth',1.5,'HandleVisibility','off');
hold off
box on; grid off
xlabel('time')
ylabel('displacement')
legend([h1 h2 h3],{'Original dynamics~',...
                   'Data-driven model~',...
                   'Training Data~'}, ...
                   'Location','Best')
xlim([t0_val t1_val])
ylim(1.15*[min(X_true(:,1)) max(X_true(:,1))])
set(gca,'LineWidth',1.2,'TickDir','in','XMinorTick','on','YMinorTick','on')

exportgraphics(fig1,fullfile(fig_dir,[case_name,'_disp.pdf']),'ContentType','vector')
exportgraphics(fig1,fullfile(fig_dir,[case_name,'_disp.png']),'Resolution',600)
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Figure 2: velocity validation
% -------------------------------------------------------------------------
fig2 = figure('Color','w',...
              'Units','inches',...
              'Position',[1 1 10 5]);
h1 =  plot(time_val  ,X_true(:,2) ,'-' ,'Color',clr_true ,'LineWidth' ,2.0);
hold on
h2 =  plot(time_val  ,X_model(:,2),'--','Color',clr_model,'LineWidth' ,1.5);
h3 =  plot(time_train,X_train(:,2),'.' ,'Color',clr_data ,'MarkerSize',22);
h4 = xline(t1_train  ,'--','Color',[0.35 0.35 0.35],...
                      'LineWidth',1.5,'HandleVisibility','off');
hold off
box on; grid off
xlabel('time')
ylabel('velocity')
legend([h1 h2 h3],{'Original dynamics~',...
                   'Data-driven model~',...
                   'Training Data~'}, ...
                   'Location','Best')
xlim([t0_val t1_val])
ylim(1.15*[min(X_true(:,2)) max(X_true(:,2))])
set(gca,'LineWidth',1.2,'TickDir','in','XMinorTick','on','YMinorTick','on')

exportgraphics(fig2,fullfile(fig_dir,[case_name,'_velo.pdf']),'ContentType','vector')
exportgraphics(fig2,fullfile(fig_dir,[case_name,'_velo.png']),'Resolution',600)
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Figure 3: phase portrait validation
% -------------------------------------------------------------------------
fig3 = figure('Color','w',...
              'Units','inches',...
              'Position',[1 1 7 6]);

h1 = plot(X_true(:,1) ,X_true(:,2) ,'-' ,'Color',clr_phase,'LineWidth' ,2.0);
hold on
h2 = plot(offset*X_model(idx_period,1),offset*X_model(idx_period,2),...
          '--','Color',clr_model,'LineWidth',1.5);
h3 = plot(X_train(:,1),X_train(:,2),'.' ,'Color',clr_data ,'MarkerSize',22);
hold off
box on; grid off
xlabel('displacement')
ylabel('velocity')
legend([h1 h2 h3],{'Original dynamics~',...
                   'Data-driven model~',...
                   'Training Data~'}, ...
                   'Location','Best')
axis equal
xlim(1.10*[min(X_true(:,1)) max(X_true(:,1))])
ylim(1.10*[min(X_true(:,2)) max(X_true(:,2))])
set(gca,'LineWidth',1.2,'TickDir','in','XMinorTick','on','YMinorTick','on')

exportgraphics(fig3,fullfile(fig_dir,[case_name,'_phase_space.pdf']),'ContentType','vector')
exportgraphics(fig3,fullfile(fig_dir,[case_name,'_phase_space.png']),'Resolution',600)
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Figure 4: validation under seen and unseen initial conditions
% -------------------------------------------------------------------------
fig4 = figure('Color','w',...
              'Units','inches',...
              'Position',[1 1 8.2 6.0]);

% Colors
clr_seen = clr_true;
clr_ic   = lines(Ntest);

% Seen initial condition
h1 = plot(X_true(:,1),X_true(:,2),'-','Color',clr_seen,'LineWidth',2.0);
hold on
h2 = plot(offset*X_model(idx_period,1),offset*X_model(idx_period,2),...
          '--','Color',clr_seen,'LineWidth',1.5);

% Training data
h3 = plot(X_train(:,1),X_train(:,2),'.','Color',clr_data,'MarkerSize',22);

% Unseen initial conditions
for j = 1:Ntest
    plot(X_true_ic{j}(:,1),X_true_ic{j}(:,2),...
         '-','Color',clr_ic(j,:),'LineWidth',2.0);

    plot(offset*X_model_ic{j}(idx_period,1),...
         offset*X_model_ic{j}(idx_period,2),...
        '--','Color',clr_ic(j,:),'LineWidth',1.5);
end

% Minimal legend handles
h4 = plot(nan,nan,'-' ,'Color',[0.25 0.25 0.25],'LineWidth',2.8);
h5 = plot(nan,nan,'--','Color',[0.25 0.25 0.25],'LineWidth',2.2);

hold off
box on; grid off
xlabel('displacement')
ylabel('velocity')

legend([h4 h5 h3],...
       {'Original dynamics~',...
        'Data-driven model~',...
        'Training Data~'},...
       'Location','eastoutside')

axis equal

% Axis limits using all displayed trajectories
x_all = [X_true(:,1); offset*X_model(idx_period,1); X_train(:,1)];
y_all = [X_true(:,2); offset*X_model(idx_period,2); X_train(:,2)];

for j = 1:Ntest
    x_all = [x_all; X_true_ic{j}(:,1); offset*X_model_ic{j}(idx_period,1)];
    y_all = [y_all; X_true_ic{j}(:,2); offset*X_model_ic{j}(idx_period,2)];
end

pad_x = 0.08*(max(x_all) - min(x_all));
pad_y = 0.08*(max(y_all) - min(y_all));

xlim([min(x_all)-pad_x max(x_all)+pad_x])
ylim([min(y_all)-pad_y max(y_all)+pad_y])

set(gca,'LineWidth',1.2,...
        'TickDir','in',...
        'XMinorTick','on',...
        'YMinorTick','on')

exportgraphics(fig4,fullfile(fig_dir,[case_name,'_seen_unseen_ICs.pdf']),'ContentType','vector')
exportgraphics(fig4,fullfile(fig_dir,[case_name,'_seen_unseen_ICs.png']),'Resolution',600)
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Figure 5: coefficient matrix heatmap
% -------------------------------------------------------------------------
fig5 = figure('Color','w',...
              'Units','inches',...
              'Position',[1 1 6.5 5]);

imagesc(XI)
colormap(redbluecmap(256))
colorbar

cmax = max(abs(XI(:)));
if cmax == 0
    cmax = 1;
end
clim([-cmax cmax])

xticks(1:2)
xticklabels({'$\dot{x}_1$','$\dot{x}_2$'})

basis_labels = MySINDyModel.GenerateCoeffList(XI,{'x_1','x_2'});
yticks(1:size(XI,1))
yticklabels(basis_labels(2:end,1))

box on; grid off
title('Sparse coefficient matrix')
set(gca,'LineWidth',1.2,'TickDir','out','FontSize',14)

exportgraphics(fig5,fullfile(fig_dir,[case_name,'_coefficients.pdf']),'ContentType','vector')
exportgraphics(fig5,fullfile(fig_dir,[case_name,'_coefficients.png']),'Resolution',600)
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Dynamical system evolution law
% -------------------------------------------------------------------------
% State definition:
%   x1 = displacement
%   x2 = velocity
% -------------------------------------------------------------------------
% Governing equations:
%   dx1/dt = x2
%   dx2/dt = -omega_n^2 x1
% -------------------------------------------------------------------------
function dsdt = dsdt_HarmonicOscillator(~,state,Param)
    
    % state coordinates
    x1 = state(1);
    x2 = state(2);

    % natural frequency
    omega_n = Param.omega_n;

    % system equations
    dx1dt = x2;
    dx2dt = -omega_n^2*x1;
    dsdt  = [dx1dt; dx2dt];
end
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
