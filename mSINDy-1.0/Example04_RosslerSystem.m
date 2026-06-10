% -------------------------------------------------------------------------
%  Example04_RosslerSystem.m
% -----------------------------------------------------------------
%  Programmer: Americo Cunha Jr
%              americo@lncc.br
%
%  Originally programmed in: May  16, 2026
%            Last update in: June 01, 2026
% -----------------------------------------------------------------
%  Sparse Identification of Nonlinear Dynamics (SINDy)
%  Example 4: Rossler chaotic system
%
%  This script uses the mSINDy class to identify the governing equations
%  of the Rossler chaotic system from sampled trajectory data.
%
%  The Rossler system is
%
%      dx1/dt = -x2 - x3,
%      dx2/dt =  x1 + a*x2,
%      dx3/dt =  b + x3*(x1 - c).
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
disp(' Example 4: Rossler chaotic system                             ')
disp('===============================================================')
disp(' ')

case_name = 'Example04_RosslerSystem';
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
t0 = 0.0;
t1 = 500.0;
dt = 0.01;

Param.a = 0.10;
Param.b = 0.10;
Param.c = 14.0;

IC = [-8.0; 8.0; 0.0];

TrueEvolutionLaw = @(t,s) dsdt_Rossler(t,s,Param);
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Training data
% -------------------------------------------------------------------------
t0_train = t0;
t1_train = 0.1*t1;
dt_train = 10*dt;
time_dense_train = (t0_train:dt_train:t1_train)';

[~,X_dense_train] = ode45(TrueEvolutionLaw,time_dense_train,IC);

% Random sparse sampling
Nsamples = 200;
Nsamples = min(Nsamples,length(time_dense_train));

idx_sample = randperm(length(time_dense_train),Nsamples);
idx_sample = sort(idx_sample);

time_train = time_dense_train(idx_sample);
X_train_clean = X_dense_train(idx_sample,:);

[Ndata,Nvars] = size(X_train_clean);

sigma = 0.05;
X_train = X_train_clean + sigma*std(X_train_clean,0,1).*randn(Ndata,Nvars);

dXdt_train = zeros(Ndata,Nvars);
for k = 1:Ndata
    dXdt_train(k,:) = TrueEvolutionLaw(0.0,X_train(k,:)')';
end
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Sparse identification with mSINDy
% -------------------------------------------------------------------------
Opts.lambda     = 5.0e-3;
Opts.Nthresh    = 10;
Opts.solver     = 'stls';
Opts.order_poly = 4;
Opts.order_trig = 0;
Opts.order_exp  = 0;
Opts.order_log  = 0;
Opts.verbose    = false;

MySINDyModel = mSINDy(Opts);

[XI,THETA] = MySINDyModel.Run(X_train,dXdt_train);
CoeffList  = MySINDyModel.GenerateCoeffList(XI,{'x_1','x_2','x_3'});

disp('Identified coefficient table:')
disp(CoeffList)

DataDrivenModel = @(t,s) MySINDyModel.EvolutionLaw(XI,t,s);
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Validation on same initial condition
% -------------------------------------------------------------------------
t0_val   = t0;
t1_val   = t1;
dt_val   = dt;
time_val = (t0_val:dt_val:t1_val)';

[~,X_true]  = ode45(TrueEvolutionLaw,time_val,IC);
[~,X_model] = ode45(DataDrivenModel ,time_val,IC);

train_mask = time_val <= t1_train;
test_mask  = time_val >  t1_train;
short_mask = time_val <= t1_train;
stat_mask  = time_val >= 0.8*t1;
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Validation on unseen initial conditions
% -------------------------------------------------------------------------
IC_test = [ 1.0, -2.0,  3.0;
           -4.0, -3.0, -6.0;
            0.5,  1.5,  0.8];

Ntest      = size(IC_test,2);
X_true_ic  = cell(Ntest,1);
X_model_ic = cell(Ntest,1);

for j = 1:Ntest
    [~,X_true_ic{j} ] = ode45(TrueEvolutionLaw,time_val,IC_test(:,j));
    [~,X_model_ic{j}] = ode45(DataDrivenModel ,time_val,IC_test(:,j));
end
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Trajectory error metrics
% -------------------------------------------------------------------------
rel_err_short = RelativeError(X_true(short_mask,:),X_model(short_mask,:));
rel_err_train = RelativeError(X_true(train_mask,:),X_model(train_mask,:));
rel_err_test  = RelativeError(X_true(test_mask,:) ,X_model(test_mask,:) );

fprintf('\n--- Pointwise trajectory errors ---\n')
fprintf('Relative error on short-time interval : %.3e\n',rel_err_short);
fprintf('Relative error on training interval   : %.3e\n',rel_err_train);
fprintf('Relative error on test interval       : %.3e\n',rel_err_test );
fprintf('\nNote: long-time pointwise errors are expected to grow in chaotic systems.\n')
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Attractor statistical metrics
% -------------------------------------------------------------------------
X_true_stat  = X_true(stat_mask,:);
X_model_stat = X_model(stat_mask,:);

Stats_true.mean  = mean(X_true_stat,1);
Stats_model.mean = mean(X_model_stat,1);

Stats_true.std  = std(X_true_stat,0,1);
Stats_model.std = std(X_model_stat,0,1);

Stats_true.cov  = cov(X_true_stat);
Stats_model.cov = cov(X_model_stat);

mean_err = RelativeError(Stats_true.mean,Stats_model.mean);
std_err  = RelativeError(Stats_true.std ,Stats_model.std );
cov_err  = RelativeError(Stats_true.cov ,Stats_model.cov );

fprintf('\n--- Attractor statistical errors ---\n')
fprintf('Mean error       : %.3e\n',mean_err)
fprintf('Std. dev. error  : %.3e\n',std_err )
fprintf('Covariance error : %.3e\n',cov_err )
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

clr_true  = [0.85 0.00 0.00];
clr_model = [0.00 0.00 0.00];
clr_data  = [0.00 0.50 0.00];
clr_phase = [0.00 0.65 0.90];
clr_train = [0.00 0.20 0.90];
clr_test  = [0.90 0.10 0.00];

offset1 = 1.00;
offset2 = 1.00;
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Figure 1: x1 complete time series
% -------------------------------------------------------------------------
fig1 = figure('Color','w','Units','inches','Position',[1 1 10 5]);

h1 = plot(time_val,X_true(:,1),'-','Color',clr_true,'LineWidth',2.0);
hold on
h2 = plot(time_val,offset2*X_model(:,1),'--','Color',clr_model,'LineWidth',1.5);
h3 = plot(time_train,X_train(:,1),'.','Color',clr_data,'MarkerSize',18);
xline(t1_train,'--','Color',[0.35 0.35 0.35],...
      'LineWidth',1.5,'HandleVisibility','off')
hold off

box on; grid off
xlabel('time')
ylabel('$x_1$')
legend([h1 h2 h3],{'Original dynamics~',...
                   'Data-driven model~',...
                   'Training Data~'},...
                   'Location','Best')
xlim([t0_val t1_val])
ylim(1.15*[min(X_true(:,1)) max(X_true(:,1))])
set(gca,'LineWidth',1.2,'TickDir','in','XMinorTick','on','YMinorTick','on')

exportgraphics(fig1,fullfile(fig_dir,[case_name,'_x1.pdf']),'ContentType','vector')
exportgraphics(fig1,fullfile(fig_dir,[case_name,'_x1.png']),'Resolution',600)
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Figure 2: x2 complete time series
% -------------------------------------------------------------------------
fig2 = figure('Color','w','Units','inches','Position',[1 1 10 5]);

h1 = plot(time_val,X_true(:,2),'-','Color',clr_true,'LineWidth',2.0);
hold on
h2 = plot(time_val,offset2*X_model(:,2),'--','Color',clr_model,'LineWidth',1.5);
h3 = plot(time_train,X_train(:,2),'.','Color',clr_data,'MarkerSize',18);
xline(t1_train,'--','Color',[0.35 0.35 0.35],...
      'LineWidth',1.5,'HandleVisibility','off')
hold off

box on; grid off
xlabel('time')
ylabel('$x_2$')
legend([h1 h2 h3],{'Original dynamics~',...
                   'Data-driven model~',...
                   'Training Data~'},...
                   'Location','Best')
xlim([t0_val t1_val])
ylim(1.15*[min(X_true(:,2)) max(X_true(:,2))])
set(gca,'LineWidth',1.2,'TickDir','in','XMinorTick','on','YMinorTick','on')

exportgraphics(fig2,fullfile(fig_dir,[case_name,'_x2.pdf']),'ContentType','vector')
exportgraphics(fig2,fullfile(fig_dir,[case_name,'_x2.png']),'Resolution',600)
% -------------------------------------------------------------------------



% -------------------------------------------------------------------------
% Figure 3: x3 complete time series
% -------------------------------------------------------------------------
fig3 = figure('Color','w','Units','inches','Position',[1 1 10 5]);

h1 = plot(time_val,X_true(:,3),'-','Color',clr_true,'LineWidth',2.0);
hold on
h2 = plot(time_val,offset2*X_model(:,3),'--','Color',clr_model,'LineWidth',1.5);
h3 = plot(time_train,X_train(:,3),'.','Color',clr_data,'MarkerSize',18);
xline(t1_train,'--','Color',[0.35 0.35 0.35],...
    'LineWidth',1.5,'HandleVisibility','off')
hold off

box on; grid off
xlabel('time')
ylabel('$x_3$')
legend([h1 h2 h3],{'Original dynamics~',...
                   'Data-driven model~',...
                   'Training Data~'},...
                   'Location','Best')
xlim([t0_val t1_val])
ylim(1.15*[min(X_true(:,3)) max(X_true(:,3))])
set(gca,'LineWidth',1.2,'TickDir','in','XMinorTick','on','YMinorTick','on')

exportgraphics(fig3,fullfile(fig_dir,[case_name,'_x3.pdf']),'ContentType','vector')
exportgraphics(fig3,fullfile(fig_dir,[case_name,'_x3.png']),'Resolution',600)
% -------------------------------------------------------------------------




% -------------------------------------------------------------------------
% Figure 3: 3D attractor validation
% -------------------------------------------------------------------------
fig4 = figure('Color','w','Units','inches','Position',[1 1 7 6]);

idx_plot = stat_mask;

h1 = plot3(X_true(idx_plot,1),X_true(idx_plot,2),X_true(idx_plot,3),...
           '-','Color',clr_phase,'LineWidth',2.0);
hold on
h2 = plotDashedCurve3D(offset1*X_model(idx_plot,1),...
                       offset1*X_model(idx_plot,2),...
                       X_model(idx_plot,3),'--',...
                       'Color',clr_model,'LineWidth',1.5);
h3 = plot3(X_train(:,1),X_train(:,2),X_train(:,3),...
           '.','Color',clr_data,'MarkerSize',18);
hold off

box on; grid on
xlabel('$x_1$')
ylabel('$x_2$')
zlabel('$x_3$')
legend([h1 h2 h3],{'Original dynamics~',...
                   'Data-driven model~',...
                   'Training Data~'},...
                   'Location','Best')
view(45,25)
set(gca,'LineWidth',1.2,'TickDir','in',...
        'XMinorTick','on','YMinorTick','on','ZMinorTick','on')

exportgraphics(fig4,fullfile(fig_dir,[case_name,'_phase_space_3D.pdf']),'ContentType','vector')
exportgraphics(fig4,fullfile(fig_dir,[case_name,'_phase_space_3D.png']),'Resolution',600)
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Figure 5: attractor comparison with equal axis limits
% -------------------------------------------------------------------------
idx_plot_true  = stat_mask;
idx_plot_model = stat_mask;

allX = [X_true(idx_plot_true,:); X_model(idx_plot_model,:)];
lims = [min(allX,[],1); max(allX,[],1)];
pad  = 0.08*(lims(2,:) - lims(1,:));
lims = [lims(1,:) - pad; lims(2,:) + pad];

fig5 = figure('Color','w','Units','inches','Position',[1 1 12 5]);

subplot(1,2,1)

h1 = plot3(X_true(idx_plot_true,1),...
           X_true(idx_plot_true,2),...
           X_true(idx_plot_true,3),...
           '-','Color',clr_phase,'LineWidth',2.0);
hold on
h2 = plot3(X_train(:,1),X_train(:,2),X_train(:,3),...
           '.','Color',clr_data,'MarkerSize',18);
hold off

box on; grid on
xlabel('$x_1$')
ylabel('$x_2$')
zlabel('$x_3$')
title('Original attractor')
legend([h1 h2],{'Original dynamics~',...
                'Training Data~'},...
                'Location','Best')
xlim(lims(:,1))
ylim(lims(:,2))
zlim(lims(:,3))
view(45,25)
set(gca,'LineWidth',1.2,...
        'TickDir','in',...
        'XMinorTick','on',...
        'YMinorTick','on',...
        'ZMinorTick','on')

subplot(1,2,2)

h3 = plot3(X_model(idx_plot_model,1),...
           X_model(idx_plot_model,2),...
           X_model(idx_plot_model,3),...
           '--','Color',clr_model,'LineWidth',1.8);

box on; grid on
xlabel('$x_1$')
ylabel('$x_2$')
zlabel('$x_3$')
title('Reconstructed attractor')
legend(h3,{'Data-driven model~'},...
           'Location','Best')
xlim(lims(:,1))
ylim(lims(:,2))
zlim(lims(:,3))
view(45,25)
set(gca,'LineWidth',1.2,...
        'TickDir','in',...
        'XMinorTick','on',...
        'YMinorTick','on',...
        'ZMinorTick','on')

exportgraphics(fig5,fullfile(fig_dir,[case_name,'_attractor_comparison.pdf']),'ContentType','vector')
exportgraphics(fig5,fullfile(fig_dir,[case_name,'_attractor_comparison.png']),'Resolution',600)
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Figure 6: marginal histograms of the attractor
% -------------------------------------------------------------------------
fig6 = figure('Color','w','Units','inches','Position',[1 1 12 4]);
tiledlayout(1,3,'Padding','compact','TileSpacing','compact')

var_labels = {'$x_1$','$x_2$','$x_3$'};

for j = 1:3
    nexttile

    histogram(X_true_stat(:,j),40,...
        'Normalization','pdf',...
        'DisplayStyle','stairs',...
        'LineWidth',2.5)

    hold on

    histogram(X_model_stat(:,j),40,...
        'Normalization','pdf',...
        'DisplayStyle','stairs',...
        'LineWidth',2.5)

    hold off

    box on; grid off
    xlabel(var_labels{j})
    ylabel('pdf')

    if j == 1
        legend({'Original dynamics~',...
                'Data-driven model~'},...
                 'Location','Best')
    end

    set(gca,'LineWidth',1.1,'TickDir','in','XMinorTick','on','YMinorTick','on')
end

exportgraphics(fig6,fullfile(fig_dir,[case_name,'_attractor_histograms.pdf']),'ContentType','vector')
exportgraphics(fig6,fullfile(fig_dir,[case_name,'_attractor_histograms.png']),'Resolution',600)
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Figure 7: validation under seen and unseen initial conditions
% -------------------------------------------------------------------------
fig7 = figure('Color','w','Units','inches','Position',[1 1 12 7]);
tiledlayout(4,2,'Padding','compact','TileSpacing','compact')

nexttile([4 1])

h1 = plot(time_val(train_mask),X_true(train_mask,1),'-',...
          'Color',clr_data,'LineWidth',2.0);
hold on
h2 = plot(time_val(test_mask),X_true(test_mask,1),'-',...
          'Color',clr_test,'LineWidth',2.0);
h3 = plot(time_val(test_mask),offset2*X_model(test_mask,1),'--',...
          'Color',clr_model,'LineWidth',1.5);
xline(t1_train,':','Color',[0.25 0.25 0.25],...
      'LineWidth',1.5,'HandleVisibility','off')
hold off

box on; grid off
title('Train/test validation')
xlabel('time')
ylabel('$x_1$')
legend([h1 h2 h3],{'Training interval',...
                   'Test interval',...
                   'Model prediction'},...
                   'Location','best')
set(gca,'LineWidth',1.1,'TickDir','in','XMinorTick','on','YMinorTick','on')

colors = [0.00 0.20 0.90;
          0.85 0.20 0.00;
          0.00 0.55 0.10;
          0.45 0.10 0.75];

IC_titles = {'Seen IC',...
             'Unseen IC 1',...
             'Unseen IC 2',...
             'Unseen IC 3'};

X_true_list  = [{X_true};  X_true_ic(:)];
X_model_list = [{X_model}; X_model_ic(:)];

for j = 1:length(X_true_list)

    nexttile

    h4 = plot(time_val(short_mask),X_true_list{j}(short_mask,1),'-',...
              'Color',colors(j,:),'LineWidth',2.0);
    hold on
    h5 = plot(time_val(short_mask),offset2*X_model_list{j}(short_mask,1),'--',...
              'Color','k','LineWidth',1.5);
    hold off

    box on; grid off
    title(IC_titles{j})

    if j == length(X_true_list)
        xlabel('time')
    else
        set(gca,'XTickLabel',[])
    end

    ylabel('$x_1$')

    if j == 1
        legend([h4 h5],{'Original dynamics',...
                        'Data-driven model'},...
                        'Location','best')
    end

    xlim([0 t1_train])

    y_data = [X_true_list{j}(short_mask,1); offset2*X_model_list{j}(short_mask,1)];
    pad_y  = 0.08*(max(y_data) - min(y_data));
    ylim([min(y_data)-pad_y max(y_data)+pad_y])

    set(gca,'LineWidth',1.1,'TickDir','in','XMinorTick','on','YMinorTick','on')
end

exportgraphics(fig7,fullfile(fig_dir,[case_name,'_validation.pdf']),'ContentType','vector')
exportgraphics(fig7,fullfile(fig_dir,[case_name,'_validation.png']),'Resolution',600)
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Figure 8: coefficient matrix heatmap with active rows only
% -------------------------------------------------------------------------
active_tol  = 1.0e-8;
active_rows = any(abs(XI) > active_tol,2);
XI_active   = XI(active_rows,:);

basis_table   = MySINDyModel.GenerateCoeffList(XI,{'x_1','x_2','x_3'});
active_labels = basis_table([false; active_rows],1);

fig8 = figure('Color','w','Units','inches','Position',[1 1 7.5 4.8]);

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

exportgraphics(fig8,fullfile(fig_dir,[case_name,'_coefficients.pdf']),'ContentType','vector')
exportgraphics(fig8,fullfile(fig_dir,[case_name,'_coefficients.png']),'Resolution',600)
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
fprintf('\nFigures written to folder: %s\n',fig_dir);
fprintf('Example 4 completed successfully.\n\n');
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
% Dynamical system evolution law
% -------------------------------------------------------------------------
% State definition:
%   x1 = x
%   x2 = y
%   x3 = z
% -------------------------------------------------------------------------
% Governing equations:
%   dx1/dt = -x2 - x3
%   dx2/dt =  x1 + a*x2
%   dx3/dt =  b - c*x3 + x1*x3
% -------------------------------------------------------------------------
function dsdt = dsdt_Rossler(~,state,Param)

    x1 = state(1);
    x2 = state(2);
    x3 = state(3);

    a = Param.a;
    b = Param.b;
    c = Param.c;

    dx1dt = -x2 - x3;
    dx2dt =  x1 + a*x2;
    dx3dt =  b - c*x3 + x1*x3;

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
% Relative Frobenius error
% -------------------------------------------------------------------------
function err = RelativeError(Xref,Xmodel)

    denom = norm(Xref,'fro');

    if denom == 0
        err = NaN;
    else
        err = norm(Xref - Xmodel,'fro')/denom;
    end
end
% -------------------------------------------------------------------------
