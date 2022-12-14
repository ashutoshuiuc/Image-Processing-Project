%% MyMainScript

tic;
%% Your code here

clc; clear;
%% QUESTION 5

% Read in the images goi1.jpg and goi2_downsampled.jpg as double arrays
im1 = imread('goi1.jpg');
im2 = imread('goi2_downsampled.jpg');
im1 = double(im1);
im2 = double(im2);

figure(1); imshow(im1/255);
figure(2); imshow(im2/255);

%% PART (a)

ones_list = ones(1, 12);

for i=1:12
   figure(3); 
   imshow(im1/255); 
   [x1(i), y1(i)] = ginput(1);
   figure(4);
   imshow(im2/255);
   [x2(i), y2(i)] = ginput(1);
end

%% PART (b)

mat1 = cat(1, cat(1, x1, y1), ones_list);
mat2 = cat(1, cat(1, x2, y2), ones_list);
aff_mat = mat2*(mat1')*inv(mat1*(mat1'));

%% PART (c)

h = size(im1,1);
w = size(im1,2);
im3 = zeros(h,w);
num_invalid_coordinates = 0;

for i = 1:h
    for j = 1:w
        arr = [i, j, 1];
        arr_inv_map = inv(aff_mat)*(arr');
        x = arr_inv_map(1);
        y = arr_inv_map(2);
        if x<1 || y<1 || x>h || y>w
            num_invalid_coordinates = num_invalid_coordinates + 1;
        else
            intensity_nearest_neighbour = calculate_nearest_neighbour(im1, x, y);
            im3(i, j) = im3(i, j) + intensity_nearest_neighbour;
        end
    end
end

figure(5); imshow(im3/255);

%% PART (d)

h = size(im1,1);
w = size(im1,2);
im4 = zeros(h,w);
num_invalid_coordinates = 0;

for i = 1:h
    for j = 1:w
        arr = [i, j, 1];
        arr_inv_map = inv(aff_mat)*(arr');
        x = arr_inv_map(1);
        y = arr_inv_map(2);
        if x<1 || y<1 || x>h || y>w
            num_invalid_coordinates = num_invalid_coordinates + 1;
        else
            intensity_bilinear_interp = calculate_bilinear_interpolation(im1, x, y);
            im4(i, j) = im4(i, j) + intensity_bilinear_interp;
        end
    end
end

figure(6); imshow(im4/255);

toc;

%% Function definitions for PART (c)

function nearest_neighbor = calculate_nearest_neighbour(im1, x, y)
% NEAREST_NEIGHBOR Computes intensity value for a pixel using nearest neighbor.
% Inputs:
%       - Image-1: im1 
%       - X from reverse warping: x
%       - Y from reverse warping: y
% Returns:
%       - Intensity value of nearest neighbour: I
    x11 = fix(x);
    y11 = fix(y)+1;
    x12 = fix(x)+1;
    y12 = fix(y)+1;
    x21 = fix(x);
    y21 = fix(y);
    x22 = fix(x)+1;
    y22 = fix(y);
    A11 = abs((x-x11)*(y11-y));
    A12 = abs((x12-x)*(y12-y));
    A21 = abs((x-x21)*(y-y21));
    A22 = abs((x22-x)*(y-y22));
    areas = [A11,A12,A21,A22];
    [~, argmax] = max(areas);
    if argmax == 1
        nearest_neighbor = im1(x22, y22);
    elseif argmax == 2
        nearest_neighbor = im1(x21, y21);
    elseif argmax == 3
        nearest_neighbor = im1(x12, y12);
    else
        nearest_neighbor = im1(x11, y11);
    end
end

%% Function definitions for PART (d)

function bilinear_interpolation = calculate_bilinear_interpolation(im1, x, y)
% BILINEAR_INTERPOLATION Computes intensity value for a pixel using interpolation value.
% Inputs:
%       - Image-1: im1 
%       - X from reverse warping: x
%       - Y from reverse warping: y
% Returns:
%       - Intensity value using bilinear interpolation: I
    x11 = fix(x);
    y11 = fix(y)+1;
    x12 = fix(x)+1;
    y12 = fix(y)+1;
    x21 = fix(x);
    y21 = fix(y);
    x22 = fix(x)+1;
    y22 = fix(y);
    A11 = abs((x-x11)*(y11-y));
    A12 = abs((x12-x)*(y12-y));
    A21 = abs((x-x21)*(y-y21));
    A22 = abs((x22-x)*(y-y22));
    areas = [A11,A12,A21,A22];
    a11 = im1(x11, y11);
    a12 = im1(x12, y12);
    a21 = im1(x21, y21);
    a22 = im1(x22, y22);
    bilinear_interpolation = (A11*a22) + (A22*a11) + (A12*a21) + (A21*a12);
end