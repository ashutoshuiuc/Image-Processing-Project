%% MyMainScript

% To run this code, the unzipped directory titled 'ORL' should be present 
% in the same directory as this code file. No modifications are needed in 
% the structure of the 'ORL' directory and its subdirectories for this 
% code to work correctly.

tic;

%% Your code here

clear; clc;

%% Main

[X_train, X_test, y_train, y_test] = load_complete_orl_data();

X_mean = mean(X_train,2);
X_scaled = X_train - X_mean;
k = 50;
[U,S,V] = svds(X_scaled/sqrt(size(X_scaled,2)-1), k);
thresholds = linspace(1e6,14e6,14);
false_positives = zeros(size(thresholds));
false_negatives = zeros(size(thresholds));
for i = 1:numel(thresholds)
    X_projected = transpose(U)*X_scaled;
    for j = 1:size(X_test,2)
        ans = transpose(U)*(X_test(:,j)-X_mean);
        temp = sum((X_projected-ans).^2, 1);
        [minimum, index] = min(temp);
        if j<=128 && minimum>thresholds(i)
            false_negatives(i) = false_negatives(i)+1;
        end
        if j>=129 && minimum<thresholds(i)
            false_positives(i) = false_positives(i)+1;
        end
    end
end
false_negative_rate = false_negatives/128;
false_positive_rate = false_positives/32;

figure(1)
plot(thresholds, false_negative_rate, '-o', 'MarkerEdgeColor', [0, 0, 0], 'MarkerFaceColor', [0.9, 0.9, 0.9], ...
    'DisplayName', 'False Negative Rate');
hold on;
plot(thresholds, false_positive_rate, '-o', 'MarkerEdgeColor', [0, 0, 0], 'MarkerFaceColor', [0.9, 0.9, 0.9], ...
    'DisplayName', 'False Positive Rate');
hold off;
xlabel('Threshold'); ylabel('Rate');
legend('Location', 'best');
grid on;
grid minor;

toc;

%% Function Definitions

function [X_train, X_test, y_train, y_test] = load_complete_orl_data()

    cd('ORL');
    orl = dir;
    orl = orl(~ismember({orl.name}, {'.', '..'}));
    flags = [orl.isdir];
    orl = orl(flags);
    X_train = [];
    X_test = [];
    y_train = [];
    y_test = [];
    for i = 1:32
        cd(orl(i).name);
        label = i;
        images = dir; 
        images = images(~ismember({images.name}, {'.', '..'}));
        for j = 1:6
            file = images(j).name;
            I = double(imread(file));
            X_train = [X_train I(:)];
            y_train = [y_train label];
        end
        for j = 7:numel(images)
            file = images(j).name;
            I = double(imread(file));
            X_test = [X_test I(:)];
            y_test = [y_test label];
        end
        cd('..');
    end
    for i = 33:numel(orl)
        cd(orl(i).name);
        label = i;
        images = dir; 
        images = images(~ismember({images.name}, {'.', '..'}));
        for j = 7:numel(images)
            file = images(j).name;
            I = double(imread(file));
            X_test = [X_test I(:)];
            y_test = [y_test label];
        end
        cd('..');
    end
    cd('..');
    
end