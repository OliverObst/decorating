load('stock_prediction.mat')
% Combine the training and test data
full_dat = [In, Eval, Test];

% Create a time vector for the full dataset
time_full = 1:length(full_dat);

% Create a time vector for the prediction, starting at step 251
time_pred = 251:length(Out);

% read LSTM prediction using dlmread, specifying the delimiter as a space
lstm = dlmread('../lstm/LSTMRESULTS/FMEprediction.csv', ' ');
% Extract the first column from the data_matrix
lstm_column = lstm(:, 1);

% read ESN prediction using dlmread
esn_column = dlmread('../esn/RESULTS/FME_prediction.csv', ',');


arima = dlmread('../arima/forecast_fre.csv', ' ');

cmap = [0.0 0.0 0.0; % Black
        1.0 0.5 0.0; % Orange
        0.8 0.2 0.2; % Red
        0.0 0.6 0.0; % Green
        0.1 0.4 0.7]; % Deep Blue

% Set up the figure and axis with a light gray background
figure('Color', [0.9 0.9 0.9]);
ax = gca;
hold on;

% Applying colormap
colormap(cmap);

% Plot the true data from step 251
plot(time_pred, full_dat(251:end), 'LineWidth', 2, 'Color', cmap(1,:), 'DisplayName', 'True Data');

% Plot the predicted data
plot(time_pred, Out(251:end), 'LineWidth', 2, 'Color', cmap(2,:), 'LineStyle', '--', 'DisplayName', 'LRNN');
% lstm
plot(time_pred, lstm_column, 'LineWidth', 2, 'Color', cmap(3,:), 'LineStyle', ':', 'DisplayName', 'LSTM');
% esn
plot(time_pred, esn_column, 'LineWidth', 2, 'Color', cmap(4,:), 'LineStyle', ':', 'DisplayName', 'ESN');
% arima
plot(time_pred, arima, 'LineWidth', 2, 'Color', cmap(5,:), 'LineStyle', ':', 'DisplayName', 'ARIMA');

% Configure the main axis
set(ax, 'FontSize', 16);
set(ax, 'LineWidth', 1.5);
set(ax, 'Box', 'on');
xlabel('Time Step', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Value', 'FontSize', 18, 'FontWeight', 'bold');
title('Stock Prediction: FME', 'FontSize', 20, 'FontWeight', 'bold');

% Add a legend
legend('show', 'Location', 'best');

% Create inset axes
inset_pos = [0.15, 0.14, 0.3, 0.22]; % [left, bottom, width, height]
ax_inset = axes('Position', inset_pos);
box on;
hold on;

% Plot the full data in the inset
plot(ax_inset, time_full, full_dat, 'LineWidth', 1.5, 'Color', cmap(1,:), 'DisplayName', 'True Data');
plot(ax_inset, time_pred, Out(251:end), 'LineWidth', 1.5, 'Color', cmap(2,:), 'LineStyle', '--', 'DisplayName', 'Predicted Data');
plot(time_pred, lstm_column, 'LineWidth', 1.5, 'Color', cmap(3,:), 'LineStyle', '-.', 'DisplayName', 'LSTM');
plot(time_pred, esn_column, 'LineWidth', 1.5, 'Color', cmap(4,:), 'LineStyle', '-.', 'DisplayName', 'ESN');
plot(time_pred, arima, 'LineWidth', 1.5, 'Color', cmap(5,:), 'LineStyle', '-.', 'DisplayName', 'ARIMA');

% Configure the inset axis
set(ax_inset, 'FontSize', 12);
set(ax_inset, 'LineWidth', 1);
xticklabels(ax_inset, []);
yticklabels(ax_inset, []);
title(ax_inset, 'Full Series Overview', 'FontSize', 14, 'FontWeight', 'bold');

% Save the figure to a high-quality image
print('-dpng', '-r300', 'timeseries_prediction_g.png');

hold off;
