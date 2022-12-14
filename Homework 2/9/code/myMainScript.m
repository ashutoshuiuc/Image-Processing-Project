%% MyMainScript

tic;
%% Your code here

clc; clear;

% read in the images LC1.png and LC2.jpg
I1 = imread('LC1.png');
I2 = imread('LC2.jpg');

% display the images LC1.png and LC2.png
figure(1); imshow(I1); colormap(gray);
figure(2); imshow(I2); colormap(gray);

% Apply global HE on LC1.png and LC2.png and display the output
I1_ghe = histeq(I1);
figure(3); imshow(I1_ghe); colormap(gray);
I2_ghe = histeq(I2);
figure(4); imshow(I2_ghe); colormap(gray);

% Change the data type to double
I1 = double(I1);
I2 = double(I2);

% Apply local HE on LC1.png and LC2.png with filter size 7x7
% and display the output
I1_lhe_A = local_histogram_equalization(I1, 7);
I2_lhe_A = local_histogram_equalization(I2, 7);
figure(5); imshow(I1_lhe_A/255); colormap(gray); 
figure(6); imshow(I2_lhe_A/255); colormap(gray);

% Apply local HE on LC1.png and LC2.png with filter size 31x31
% and display the output
I1_lhe_B = local_histogram_equalization(I1, 31);
I2_lhe_B = local_histogram_equalization(I2, 31);
figure(7); imshow(I1_lhe_B/255); colormap(gray); 
figure(8); imshow(I2_lhe_B/255); colormap(gray);

% Apply local HE on LC1.png and LC2.png with filter size 51x51
% and display the output
I1_lhe_C = local_histogram_equalization(I1, 51);
I2_lhe_C = local_histogram_equalization(I2, 51);
figure(9); imshow(I1_lhe_C/255); colormap(gray); 
figure(10); imshow(I2_lhe_C/255); colormap(gray);

% Apply local HE on LC1.png and LC2.png with filter size 71x71
% and display the output
I1_lhe_D = local_histogram_equalization(I1, 71);
I2_lhe_D = local_histogram_equalization(I2, 71);
figure(11); imshow(I1_lhe_D/255); colormap(gray); 
figure(12); imshow(I2_lhe_D/255); colormap(gray); 

toc;

%% Function definitions

function calc_pdf = calculate_pdf(img)
% CALCULATE_PDF Computes Probability Density Function of an image.
% Inputs:
%       - Image: img
% Returns:
%       - Probability Density Function of img: pdf
    calc_pdf = zeros(1, 256);
    h = size(img,1);
    w = size(img,2);
    for i = 1:h
        for j = 1:w
            x = img(i, j)+1;
            calc_pdf(x) = calc_pdf(x) + 1;
        end
    end
    calc_pdf = calc_pdf/(sum(calc_pdf, "all"));
end


function lhe = local_histogram_equalization(img, window_size)
% LOCAL_HISTOGRAM_EQUALIZATION Computes Histogram Equalization of an image.
% Inputs:
%       - Image: img
%       - Size of the local histogram equalization: window_size
% Returns:
%       - Local histogram equalized image: lhe
    pad_size = fix(window_size/2);
    img_padded = padarray(img,[pad_size pad_size],0,'both');
    lhe = zeros(size(img));
    h = size(img, 1);
    w = size(img, 2);
    for i = 1:h
        for j = 1:w
            matrix_of_consideration = img_padded(i:i+window_size-1, j:j+window_size-1);
            local_pdf = calculate_pdf(matrix_of_consideration);
            lhe(i, j) = round(255*sum(local_pdf(1:img(i, j)+1), 'all'));
        end
    end
end