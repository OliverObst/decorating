%% uncomment to load previously saved run of compute_prediction
% load timeseries_prediction.mat

% Combine the training and test data
full_dat = [traindat(21,:), restdat(21,:)];

% Create a time vector for the full dataset
time_full = 1:length(full_dat);

% Create a time vector for the prediction, starting at step 251
time_pred = 251:length(Out);

% read LSTM prediction using dlmread, specifying the delimiter as a space
lstm = dlmread('../lstm/RESULTS-20230425/outputs-00.csv', ',', 1, 0);

% Extract the first column from the daarimta_matrix
lstm_column = lstm(:, 2);

% read LSTM prediction using dlmread
esn = dlmread('../esn/esnresults00.csv', ',', 1, 0);

% Extract the first column from the daarimta_matrix
esn_column = esn(:, 2);


% arima
arima = dlmread('../arima/forecast_testing_s1.csv', ' ', 1, 0);
arima = arima(1:50)

cmap = [0.1 0.4 0.7; % Deep Blue
        0.8 0.2 0.2; % Red
        0.0 0.6 0.0; % Green
        0.6 0.3 0.2; % Brown
        0.0 0.0 0.0]; % Black

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
plot(time_pred, lstm_column, 'LineWidth', 2, 'Color', cmap(3,:), 'LineStyle', '--', 'DisplayName', 'LSTM');

% esn
plot(time_pred, esn_column, 'LineWidth', 2, 'Color', cmap(4,:), 'LineStyle', '--', 'DisplayName', 'ESN');

% arima
plot(time_pred, arima, 'LineWidth', 2, 'Color', cmap(5,:), 'LineStyle', '--', 'DisplayName', 'ARIMA');

% Configure the main axis
set(ax, 'FontSize', 16);
set(ax, 'LineWidth', 1.5);
set(ax, 'Box', 'on');
xlabel('Time Step', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Value', 'FontSize', 18,'FontWeight', 'bold');
title('MSO8 Example Prediction', 'FontSize', 20, 'FontWeight', 'bold');

% Add a legend
legend('show', 'Location', 'best');

% Create inset axes
inset_pos = [0.15, 0.14, 0.3, 0.18]; % [left, bottom, width, height]
ax_inset = axes('Position', inset_pos);
box on;
hold on;

% Plot the full data in the inset
plot(ax_inset, time_full, full_dat, 'LineWidth', 1.5, 'Color', cmap(1,:), 'DisplayName', 'True Data');
plot(ax_inset, time_pred, Out(251:end), 'LineWidth', 1.5, 'Color', cmap(2,:), 'LineStyle', '--', 'DisplayName', 'LRNN');
plot(ax_inset, time_pred, lstm_column, 'LineWidth', 1.5, 'Color', cmap(3,:), 'LineStyle', '--', 'DisplayName', 'LSTM');
plot(ax_inset, time_pred, lstm_column, 'LineWidth', 1.5, 'Color', cmap(4,:), 'LineStyle', '--', 'DisplayName', 'ESN');
plot(ax_inset, time_pred, arima, 'LineWidth', 1.5, 'Color', cmap(5,:), 'LineStyle', '--', 'DisplayName', 'ARIMA');

% Configure the inset axis
set(ax_inset, 'FontSize', 10);
set(ax_inset, 'LineWidth', 1);
xticklabels(ax_inset, []);
yticklabels(ax_inset, []);
% xlabel(ax_inset, 'Time Step', 'FontSize', 12);
% ylabel(ax_inset, 'Value', 'FontSize', 12);
title(ax_inset, 'Full Series Overview', 'FontSize', 12);
% xlim(ax_inset, [1, length(full_dat)]);

% Save the figure to a high-quality image
print('-dpng', '-r300', 'timeseries_prediction.png');

hold off;
