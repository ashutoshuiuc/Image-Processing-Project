%% MyMainScript

% To run this code, the unzipped directories titled 'ORL' and 'CroppedYale'
% should be present in the same directory as this code file. No
% modifications are needed in the structure of the 'ORL' and 'CroppedYale'
% directories and its subdirectories for this code to work correctly.

tic;

%% Your code here

clear; clc;

%% Face Recognition on ORL Database (SVD routine)

disp('Face Recognition on ORL Database (SVD routine)')

[X_train, X_test, y_train, y_test] = load_data("ORL");
disp('Data Loaded')

X_mean = mean(X_train,2);
X_scaled = X_train - X_mean;
[U,S,V] = svds(X_scaled/sqrt(size(X_scaled,2)-1), 170);
disp('Eigenvectors and Eigenvalues Calculated')

k = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 170];
recognition_rates = zeros([1,numel(k)]);
for i = 1:numel(k)
    disp(['Evaluating Recognition Rate for k = ', int2str(k(i))]);
    Uk = U(:,1:k(i));
    X_projected = transpose(Uk)*X_scaled;
    for j = 1:size(X_test,2)
        ans = transpose(Uk)*(X_test(:,j)-X_mean);
        temp = sum((X_projected-ans).^2, 1);
        [minimum, index] = min(temp);
        true = y_test(j);
        predicted = y_train(index);
        if true == predicted
            recognition_rates(i) = recognition_rates(i) + 1;
        end
    end
end
recognition_rates = recognition_rates / size(X_test,2);

figure(1)
plot(k, recognition_rates, '-o', 'MarkerEdgeColor', [0, 0, 0], 'MarkerFaceColor', [0.9, 0.9, 0.9]); 
xlabel('k'); ylabel('Recognition Rate');
grid on;
grid minor;

%% Face Recognition on ORL Database (EIG routine)

disp(' ')
disp('Face Recognition on ORL Database (EIG routine)')

[X_train, X_test, y_train, y_test] = load_data("ORL");
disp('Data Loaded')

X_mean = mean(X_train,2);
X_scaled = X_train - X_mean;
[V,D] = eig(transpose(X_scaled)*X_scaled/(size(X_scaled,2)-1), 'vector');
[D, ind] = sort(D, 'descend');
V = V(:, ind);
U = X_scaled*V;
U = normc(U);
disp('Eigenvectors and Eigenvalues Calculated')

k = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 170];
recognition_rates = zeros([1,numel(k)]);
for i = 1:numel(k)
    disp(['Evaluating Recognition Rate for k = ', int2str(k(i))]);
    Uk = U(:,1:k(i));
    X_projected = transpose(Uk)*X_scaled;
    for j = 1:size(X_test,2)
        ans = transpose(Uk)*(X_test(:,j)-X_mean);
        temp = sum((X_projected-ans).^2, 1);
        [minimum, index] = min(temp);
        true = y_test(j);
        predicted = y_train(index);
        if true == predicted
            recognition_rates(i) = recognition_rates(i) + 1;
        end
    end
end
recognition_rates = recognition_rates / size(X_test,2);

figure(2)
plot(k, recognition_rates, '-o', 'MarkerEdgeColor', [0, 0, 0], 'MarkerFaceColor', [0.9, 0.9, 0.9]); 
xlabel('k'); ylabel('Recognition Rate');
grid on;
grid minor;

%% Face Recognition on Yale Database (SVD routine) with top 3 eigenvalues included

disp(' ')
disp('Face Recognition on Yale Database (SVD routine) with top 3 eigenvalues included')

[X_train, X_test, y_train, y_test] = load_data("Yale");
disp('Data Loaded')

X_mean = mean(X_train,2);
X_scaled = X_train - X_mean;
[U,S,V] = svds(X_scaled/sqrt(size(X_scaled,2)-1), 1000);
disp('Eigenvectors and Eigenvalues Calculated')

k = [1, 2, 3, 5, 10, 15, 20, 30, 50, 60, 65, 75, 100, 200, 300, 500, 1000];
recognition_rates = zeros([1,numel(k)]);
for i = 1:numel(k)
    disp(['Evaluating Recognition Rate for k = ', int2str(k(i))]);
    Uk = U(:,1:k(i));
    X_projected = transpose(Uk)*X_scaled;
    for j = 1:size(X_test,2)
        ans = transpose(Uk)*(X_test(:,j)-X_mean);
        temp = sum((X_projected-ans).^2, 1);
        [minimum, index] = min(temp);
        true = y_test(j);
        predicted = y_train(index);
        if true == predicted
            recognition_rates(i) = recognition_rates(i) + 1;
        end
    end
end
recognition_rates = recognition_rates / size(X_test,2);

figure(3)
plot(k, recognition_rates, '-o', 'MarkerEdgeColor', [0, 0, 0], 'MarkerFaceColor', [0.9, 0.9, 0.9]); 
xlabel('k'); ylabel('Recognition Rate');
grid on;
grid minor;

%% Face Recognition on Yale Database (SVD routine) with top 3 eigenvalues excluded

disp(' ')
disp('Face Recognition on Yale Database (SVD routine) with top 3 eigenvalues excluded')

[X_train, X_test, y_train, y_test] = load_data("Yale");
disp('Data Loaded')

X_mean = mean(X_train,2);
X_scaled = X_train - X_mean;
[U,S,V] = svds(X_scaled/sqrt(size(X_scaled,2)-1), 1003);
disp('Eigenvectors and Eigenvalues Calculated')

k = [1, 2, 3, 5, 10, 15, 20, 30, 50, 60, 65, 75, 100, 200, 300, 500, 1000];
recognition_rates = zeros([1,numel(k)]);
for i = 1:numel(k)
    disp(['Evaluating Recognition Rate for k = ', int2str(k(i))]);
    Uk = U(:,4:k(i)+3);
    X_projected = transpose(Uk)*X_scaled;
    for j = 1:size(X_test,2)
        ans = transpose(Uk)*(X_test(:,j)-X_mean);
        temp = sum((X_projected-ans).^2, 1);
        [minimum, index] = min(temp);
        true = y_test(j);
        predicted = y_train(index);
        if true == predicted
            recognition_rates(i) = recognition_rates(i) + 1;
        end
    end
end
recognition_rates = recognition_rates / size(X_test,2);

figure(4)
plot(k, recognition_rates, '-o', 'MarkerEdgeColor', [0, 0, 0], 'MarkerFaceColor', [0.9, 0.9, 0.9]); 
xlabel('k'); ylabel('Recognition Rate');
grid on;
grid minor;

toc;

%% Function Definitions

function [X_train, X_test, y_train, y_test] = load_data(database)

    if database == "ORL"
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
        cd('..');
    end
    
    if database == "Yale"
        cd('CroppedYale');
        yale = dir;
        yale = yale(~ismember({yale.name}, {'.', '..'}));
        flags = [yale.isdir];
        yale = yale(flags);
        X_train = [];
        X_test = [];
        y_train = [];
        y_test = [];
        for i = 1:numel(yale)
            cd(yale(i).name);
            label  = yale(i).name(6:7);
            images = dir; 
            images = images(~ismember({images.name}, {'.', '..'}));
            for j = 1:40
                file = images(j).name;
                I = double(imread(file));
                X_train = [X_train I(:)];
                y_train = [y_train str2double(label)];
            end
            for j = 41:numel(images)
                file = images(j).name;
                I = double(imread(file));
                X_test = [X_test I(:)];
                y_test = [y_test str2double(label)];
            end
            cd('..');
        end
        cd('..');
    end
    
end