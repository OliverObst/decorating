% Combine the training and test data
full_dat = [traindat(1,:), restdat(1,:)];

% Create a time vector for the full dataset
time_full = 1:length(full_dat);

% Create a time vector for the prediction, starting at step 251
time_pred = 251:length(Out);

% Set up the figure and axis with a white background
figure('Color', 'w');
ax = gca;
hold on;

% Plot the true data from step 251
plot(time_pred, full_dat(251:end), 'LineWidth', 2, 'Color', 'b', 'DisplayName', 'True Data');

% Plot the predicted data
plot(time_pred, Out(251:end), 'LineWidth', 2, 'Color', 'r', 'LineStyle', '--', 'DisplayName', 'Predicted Data');

% Configure the main axis
set(ax, 'FontSize', 14);
set(ax, 'LineWidth', 1.5);
set(ax, 'Box', 'on');
xlabel('Time Step', 'FontSize', 16);
ylabel('Value', 'FontSize', 16);
title('MSO8 Example Prediction', 'FontSize', 18);

% Add a legend
legend('show', 'Location', 'best');

% Create inset axes
inset_pos = [0.15, 0.64, 0.3, 0.22]; % [left, bottom, width, height]
ax_inset = axes('Position', inset_pos);
box on;
hold on;

% Plot the full data in the inset
plot(ax_inset, time_full, full_dat, 'LineWidth', 1.5, 'Color', 'b', 'DisplayName', 'True Data');
plot(ax_inset, time_pred, Out(251:end), 'LineWidth', 1.5, 'Color', 'r', 'LineStyle', '--', 'DisplayName', 'Predicted Data');

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
