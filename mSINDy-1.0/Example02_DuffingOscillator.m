% -------------------------------------------------------------------------
%  Example02_DuffingOscillator.m
% -----------------------------------------------------------------
%  Programmer: Americo Cunha Jr
%              americo@lncc.br
%
%  Originally programmed in: May 16, 2026
%            Last update in: May 27, 2026
% -----------------------------------------------------------------
%  Sparse Identification of Nonlinear Dynamics (SINDy)
%  Example 2: Forced Duffing oscillator in autonomous lifted form
%
%  This script uses the mSINDy class to identify the governing
%  equations of a periodically forced Duffing oscillator from 
%  sampled trajectory data.
%
%  The autonomous system is
%
%      dx1/dt = x2
%      dx2/dt = -delta*x2 - alpha*x1 - beta*x1^3 + gamma*cos(x3)
%      dx3/dt = omega
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
disp(' Example 2: Forced Duffing oscillator                          ')
disp('===============================================================')
disp(' ')

case_name = 'Example02_DuffingOscillator';
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
t0          = 0.0;        % initial time
t1          = 80.0;       % final time
dt          = 0.1;        % time step
Param.delta = 0.30;       % viscous damping coefficient
Param.alpha = -1.0;       % linear stiffness coefficient
Param.beta  =  1.0;       % cubic stiffness coefficient
Param.gamma =  2.0;       % forcing amplitude
Param.omega =  2.0;       % forcing frequency
Param.x1_0  =  3.0;       % initial condition for x1
Param.x2_0  = -2.0;       % initial condition for x2
Param.x3_0  =  0.0;       % initial condition for x3

% initial conditions
IC = [Param.x1_0; Param.x2_0; Param.x3_0];

% dynamical system evolution law
TrueEvolutionLaw = @(t,s) dsdt_Duffing(t,s,Param);
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Training data
% -------------------------------------------------------------------------
t0_train   = t0;
t1_train   = 0.2*t1;
dt_train   = dt;
time_train = (t0_train:dt_train:t1_train)';

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
    dXdt_train(k,:) = TrueEvolutionLaw(0.0,X_train(k,:)')';
end
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Sparse identification with mSINDy
% -------------------------------------------------------------------------
Opts.lambda     = 0.05;
Opts.Nthresh    = 15;
Opts.solver     = 'stls';
Opts.order_poly = 5;
Opts.order_trig = 3;
Opts.order_exp  = 0;
Opts.order_log  = 0;
Opts.verbose    = true;

MySINDyModel = mSINDy(Opts);

[XI,THETA] = MySINDyModel.Run(X_train,dXdt_train);
CoeffList  = MySINDyModel.GenerateCoeffList(XI,{'x1','x2','x3'});

% Display identified coefficient table
disp('Identified coefficient table:')
disp(CoeffList)

% Data-driven evolution law
DataDrivenModel = @(t,s) MySINDyModel.EvolutionLaw(XI,t,s);
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Validation 1: train/test split on the same trajectory
% -------------------------------------------------------------------------
t0_val   = t0;
t1_val   = t1;
dt_val   = 0.1*dt;
time_val = (t0_val:dt_val:t1_val)';

[~,X_true]  = ode45(TrueEvolutionLaw,time_val,IC);
[~,X_model] = ode45(DataDrivenModel ,time_val,IC);

% Training data shown only over the training interval
train_mask = time_val <= t1_train;
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Validation 2: unseen initial conditions
% -------------------------------------------------------------------------
IC_test = [ 1.0, 1.0, 1.5;
            1.5, 1.0, 1.5;
            0.0, 0.0, 0.0];

Ntest      = size(IC_test,2);
X_true_ic  = cell(Ntest,1);
X_model_ic = cell(Ntest,1);

for j = 1:Ntest
    [~,X_true_ic{j} ] = ode45(TrueEvolutionLaw,time_val,IC_test(:,j));
    [~,X_model_ic{j}] = ode45(DataDrivenModel, time_val,IC_test(:,j));
end
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Error metrics evaluated only on the physical variables x1 and x2
% -------------------------------------------------------------------------
rel_err_train = RelativeError(X_true( train_mask,:),X_model( train_mask,:));
rel_err_test  = RelativeError(X_true(~train_mask,:),X_model(~train_mask,:));

fprintf('\nRelative error on training interval : %.3e\n',rel_err_train);
fprintf('Relative error on test interval     : %.3e\n'  ,rel_err_test );
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

% Color palette
clr_true  = [0.85 0.00 0.00];  % red
clr_model = [0.00 0.00 0.00];  % black
clr_data  = [0.00 0.50 0.00];  % green
clr_phase = [0.00 0.65 0.90];  % cyan
clr_train = [0.00 0.20 0.90];  % blue
clr_test  = [0.90 0.10 0.00];  % red-orange


% Small visual offset for data-driven curves
offset1 = 0.98;
offset2 = 1.05;

% Number of points corresponding to one oscillation period
T_period   = 2*pi/Param.omega;
Nperiod    = find(time_val <= time_val(1) + T_period,1,'last');
idx_period = 1:Nperiod;
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Figure 1: displacement validation
% -------------------------------------------------------------------------
fig1 = figure('Color','w',...
              'Units','inches',...
              'Position',[1 1 10 5]);
h1 = plot(time_val  ,X_true(:,1) ,'-' ,'Color',clr_true ,'LineWidth',2.0);
hold on
h2 = plot(time_val  ,offset2*X_model(:,1),'--','Color',clr_model,'LineWidth',1.5);
h3 = plot(time_train,X_train(:,1),'.' ,'Color',clr_data ,'MarkerSize',18);
h4 = xline(t1_train,'--','Color',[0.35 0.35 0.35],...
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
h1 = plot(time_val  ,X_true(:,2) ,'-' ,'Color',clr_true ,'LineWidth',2.0);
hold on
h2 = plot(time_val  ,offset2*X_model(:,2),'--','Color',clr_model,'LineWidth',1.5);
h3 = plot(time_train,X_train(:,2),'.' ,'Color',clr_data ,'MarkerSize',18);
h4 = xline(t1_train,'--','Color',[0.35 0.35 0.35],...
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
h1 = plot(X_true(:,1),X_true(:,2),'-','Color',clr_phase,'LineWidth',2.0);
hold on
h2 = plotDashedCurve(offset1*X_model(:,1),offset1*X_model(:,2),'--',...
                     'Color',clr_model,'LineWidth',1.5);
h3 = plot(X_train(:,1),X_train(:,2),'.','Color',clr_data,'MarkerSize',18);
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
% Figure 4: 3D phase portrait validation
% -------------------------------------------------------------------------
fig4 = figure('Color','w',...
    'Units','inches',...
    'Position',[1 1 7 6]);

h1 = plot3(X_true(:,1),X_true(:,2),X_true(:,3),'-','Color',clr_phase,'LineWidth',2.0);
hold on
h2 = plotDashedCurve3D(offset1*X_model(:,1),offset1*X_model(:,2),X_model(:,3),'--','Color',clr_model,'LineWidth',1.5);
h3 = plot3(X_train(:,1),X_train(:,2),X_train(:,3),'.','Color',clr_data,'MarkerSize',18);
hold off

box on; grid on
xlabel('displacement')
ylabel('velocity')
zlabel('forcing phase')
legend([h1 h2 h3],{'Original dynamics~',...
                   'Data-driven model~',...
                   'Training Data~'},...
                   'Location','Best')

view(45,25)
set(gca,'LineWidth',1.2,...
    'TickDir','in',...
    'XMinorTick','on',...
    'YMinorTick','on',...
    'ZMinorTick','on')

exportgraphics(fig4,fullfile(fig_dir,[case_name,'_phase_space_3D.pdf']),'ContentType','vector')
exportgraphics(fig4,fullfile(fig_dir,[case_name,'_phase_space_3D.png']),'Resolution',600)
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Figure 5: validation under seen and unseen initial conditions
% -------------------------------------------------------------------------
fig5 = figure('Color','w',...
              'Units','inches',...
              'Position',[1 1 12 7]);
tiledlayout(4,2,'Padding','compact','TileSpacing','compact')

% Left panel: train/test validation
nexttile([4 1])

h1 = plot(time_val(train_mask) ,X_true(train_mask,1)  ,'-' ,...
          'Color',clr_data ,'LineWidth',2.0);
hold on
h2 = plot(time_val(~train_mask),X_true(~train_mask,1) ,'-' ,...
          'Color',clr_test ,'LineWidth',2.0);
h3 = plot(time_val(~train_mask),offset2*X_model(~train_mask,1),'--',...
          'Color',clr_model,'LineWidth',1.5);

xline(t1_train,':','Color',[0.25 0.25 0.25],'LineWidth',1.5,'HandleVisibility','off')

hold off
box on; grid off
title('Train/test validation')
xlabel('time')
ylabel('displacement')
legend([h1 h2 h3],{'Training interval~',...
                   'Test interval~',...
                   'Model prediction~'},...
                   'Location','best')
set(gca,'LineWidth',1.1,'TickDir','in','XMinorTick','on','YMinorTick','on')


% Right panels: seen and unseen initial conditions
colors = [0.00 0.20 0.90;
          0.85 0.20 0.00;
          0.00 0.55 0.10;
          0.45 0.10 0.75];

IC_titles = {'Seen IC~',...
             'Unseen IC 1~',...
             'Unseen IC 2~',...
             'Unseen IC 3~'};

X_true_list  = [{X_true};  X_true_ic(:)];
X_model_list = [{X_model}; X_model_ic(:)];

for j = 1:length(X_true_list)

    nexttile

    h4 = plot(time_val,X_true_list{j}(:,1),'-',...
              'Color',colors(j,:),'LineWidth',2.0);
    hold on
    h5 = plot(time_val,offset2*X_model_list{j}(:,1),'--',...
              'Color','k','LineWidth',1.5);
    hold off

    box on; grid off
    title(IC_titles{j})

    if j == length(X_true_list)
        xlabel('time')
    else
        set(gca,'XTickLabel',[])
    end

    ylabel('disp.')

    if j == 1
        legend([h4 h5],{'Original dynamics~',...
                        'Data-driven model~'},...
                        'Location','best')
    end

    xlim([t0_val t1_val])

    y_data = [X_true_list{j}(:,1); offset1*X_model_list{j}(:,1)];
    pad_y  = 0.08*(max(y_data) - min(y_data));

    ylim([min(y_data)-pad_y max(y_data)+pad_y])

    set(gca,'LineWidth',1.1,...
            'TickDir','in',...
            'XMinorTick','on',...
            'YMinorTick','on')
end

exportgraphics(fig5,fullfile(fig_dir,[case_name,'_validation.pdf']),'ContentType','vector')
exportgraphics(fig5,fullfile(fig_dir,[case_name,'_validation.png']),'Resolution',600)
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Figure 6: coefficient matrix heatmap with active rows only
% -------------------------------------------------------------------------
active_tol  = 1.0e-8;
active_rows = any(abs(XI) > active_tol,2);
XI_active   = XI(active_rows,:);
basis_table = MySINDyModel.GenerateCoeffList(XI,{'x_1','x_2','x_3'});
active_labels = basis_table([false; active_rows],1);

fig6 = figure('Color','w','Units','inches','Position',[1 1 7.5 4.8]);
imagesc(XI_active)
colormap(redbluecmap(256))
colorbar
cmax = max(abs(XI_active(:)));
if cmax == 0
    cmax = 1;
end
clim([-cmax cmax])
xticks(1:3)
xticklabels({'$\dot{x}_1$','$\dot{x}_2$','$\dot{x}_3$'})
yticks(1:size(XI_active,1))
yticklabels(active_labels)
title('Active sparse coefficient matrix')
set(gca,'LineWidth',1.1,'TickDir','out','FontSize',14)
exportgraphics(fig6,fullfile(fig_dir,[case_name,'_coefficients.pdf']),'ContentType','vector')
exportgraphics(fig6,fullfile(fig_dir,[case_name,'_coefficients.png']),'Resolution',600)

fprintf('\nFigures written to folder: %s\n',fig_dir);
fprintf('Example 2 completed successfully.\n\n');
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Dynamical system evolution law
% -------------------------------------------------------------------------
% State definition:
%   x1 = displacement
%   x2 = velocity
%   x3 = omega*t
% -------------------------------------------------------------------------
% Governing equations:
%   dx1/dt = x2
%   dx2/dt = -delta*x2 - alpha*x1 - beta*x1^3 + gamma*cos(x3)
%   dx3/dt = omega
% -------------------------------------------------------------------------
function dsdt = dsdt_Duffing(~,state,Param)
    x1 = state(1);  % displacement
    x2 = state(2);  % velocity
    x3 = state(3);  % omega*t

    delta = Param.delta;
    alpha = Param.alpha;
    beta  = Param.beta;
    gamma = Param.gamma;
    omega = Param.omega;

    dx1dt = x2;
    dx2dt = -delta*x2 - alpha*x1 - beta*x1^3 + gamma*cos(x3);
    dx3dt = omega;

    dsdt = [dx1dt; dx2dt; dx3dt];
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


% -------------------------------------------------------------------------
% Function to plot a dashed curve in 3D phase space
% -------------------------------------------------------------------------
function h = plotDashedCurve3D(x,y,z,varargin)

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
z_plot = z;

x_plot(~draw_mask) = NaN;
y_plot(~draw_mask) = NaN;
z_plot(~draw_mask) = NaN;

h = plot3(x_plot,y_plot,z_plot,varargin{:});

end
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Relative Frobenius trajectory error
% -------------------------------------------------------------------------
function err = RelativeError(Xref,Xmodel)
err = norm(Xref - Xmodel,'fro')/norm(Xref,'fro');
end
% -------------------------------------------------------------------------