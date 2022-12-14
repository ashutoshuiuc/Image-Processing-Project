%% MyMainScript

% To run this code, the unzipped directory titled 'ORL' should be present 
% in the same directory as this code file. No modifications are needed in 
% the structure of the 'ORL' directory and its subdirectories for this 
% code to work correctly.

% To properly view the generated figures, do so after maximising the 
% respective MATLAB figure windows.

tic;

%% Your code here

clear; clc;

%% Reconstruction of a Face

[X_train, X_test, y_train, y_test] = load_orl_data();

file = 'ORL\s1\10.pgm';
I = double(imread(file));
x = I(:);
X_mean = mean(X_train,2);

X_scaled = X_train - X_mean;
[U,S,V] = svds(X_scaled/sqrt(size(X_scaled,2)-1), 175);
k = [2, 10, 20, 50, 75, 100, 125, 150, 175];
t = tiledlayout(2,5);
figure(1);
t.TileSpacing = 'compact';
t.Padding = 'compact';
for i = 1:numel(k)
    nexttile;
    Uk = U(:,1:k(i));
    coeffs = transpose(Uk)*(x-X_mean);
    reconstructed = sum(Uk.*transpose(coeffs), 2);
    out = reconstructed+X_mean;
    output = (out-min(out))/(max(out)-min(out));
    imshow(reshape(output,[112,92]))
    label = sprintf('k = %d', k(i));
    title(label, 'FontSize', 16);
end
nexttile;
output = (x-min(x))/(max(x)-min(x));
imshow(reshape(output,[112,92]));
label = sprintf('Original');
title(label, 'FontSize', 16);
    
%% Eigenfaces corresponding to the 25 largest eigenvalues

[X_train, X_test, y_train, y_test] = load_orl_data();

X_mean = mean(X_train,2);
X_scaled = X_train - X_mean;
[U,S,V] = svds(X_scaled/sqrt(size(X_scaled,2)-1), 25);
figure(2);
t = tiledlayout(3,9);
t.TileSpacing = 'tight';
t.Padding = 'tight';
for i = 1:25
    nexttile
    out = U(:,i);
    output = (out-min(out))/(max(out)-min(out));
    imshow(reshape(output,[112,92]));
    label = sprintf('Eigenvalue %d = %d', i, round(S(i,i)));
    title(label, 'FontSize', 10);
end

toc;

%% Function Definitions

function [X_train, X_test, y_train, y_test] = load_orl_data()

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