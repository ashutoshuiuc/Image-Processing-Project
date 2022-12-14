%% MyMainScript

tic;
%% Your code here

clc; clear;

% read in the images barbara256.png and kodak24.png as double arrays
I1 = double(imread('barbara256.png'));
I2 = double(imread('kodak24.png'));

% Add gaussian noise with 0 mean and 5 standard deviation to both
% images

I1 = I1 + normrnd(0, 5, size(I1));
I2 = I2 + normrnd(0, 5, size(I2));

% % Uncomment below to add gaussian noise with 0 mean and 10 standard 
% % deviation to both images
% 
% I1 = I1 + normrnd(0, 10, size(I1));
% I2 = I2 + normrnd(0, 10, size(I2));

% display the images I1 and I2 with noise added in them
figure(1); imshow(I1/255); colormap(gray); 
figure(2); imshow(I2/255); colormap(gray);

% apply the bilateral filter with parameters sigma_s=2 and sigma_r=2
% to both the images I1 and I2 and display the output
I1_filtered_A = mybilateralfilter(I1, 2, 2);
I2_filtered_A = mybilateralfilter(I2, 2, 2);
figure(3); imshow(I1_filtered_A/255); colormap(gray); 
figure(4); imshow(I2_filtered_A/255); colormap(gray);

% apply the bilateral filter with parameters sigma_s=0.1 and sigma_r=0.1
% to both the images I1 and I2 and display the output
I1_filtered_B = mybilateralfilter(I1, 0.1, 0.1);
I2_filtered_B = mybilateralfilter(I2, 0.1, 0.1);
figure(5); imshow(I1_filtered_B/255); colormap(gray); 
figure(6); imshow(I2_filtered_B/255); colormap(gray);

% apply the bilateral filter with parameters sigma_s=3 and sigma_r=15
% to both the images I1 and I2 and display the output
I1_filtered_C = mybilateralfilter(I1, 3, 15);
I2_filtered_C = mybilateralfilter(I2, 3, 15);
figure(7); imshow(I1_filtered_C/255); colormap(gray); 
figure(8); imshow(I2_filtered_C/255); colormap(gray);

toc;

%% Function definitions 

function filtered_img = mybilateralfilter(img, sigma_s, sigma_r)
% Does bilateral filtering on the input image using the given sigma_s and
% sigma_r values
% Inputs:
%       - Image: img
%       - Standard deviation for space based weights: sigma_s
%       - Standard deviation for intensity based weights: sigma_r
% Returns:
%       - Image filtered using bilateral filter: filtered_img
    img2 = img + 1;
    window_size = ceil(6*max(sigma_s, sigma_r));
    if rem(window_size, 2) == 0
        window_size = window_size + 1;
    end
    
    filter_matrix1 = zeros([window_size window_size]);
    for i = 1:window_size
        for j = 1:window_size
            vec1 = [i j];
            vec2 = [fix(window_size/2)+1 fix(window_size/2)+1];
            filter_matrix1(i, j) = filter_matrix1(i, j) + norm(vec1 - vec2);
        end
    end
    
    space_weights = exp(-(filter_matrix1.*filter_matrix1)/(2*sigma_s*sigma_s))/(sigma_s*(2*pi)^0.5);
    
    pad_size = fix(window_size/2);
    img_padded = padarray(img,[pad_size pad_size],0,'both');
    img_padded2 = padarray(img2,[pad_size pad_size],0,'both');
    mask = (img_padded2>0);
    filtered_img = zeros(size(img));
    h = size(img, 1);
    w = size(img, 2);
        for i = 1:h
            for j = 1:w
                matrix_of_consideration = img_padded(i:i+window_size-1, j:j+window_size-1);
                mask_moc = mask(i:i+window_size-1, j:j+window_size-1);
                filter_matrix2 = abs(matrix_of_consideration - matrix_of_consideration(pad_size+1, pad_size+1));
                intensity_weights = exp(-(filter_matrix2.*filter_matrix2)/(2*sigma_r*sigma_r))/(sigma_r*(2*pi)^0.5);
                weights = space_weights.*intensity_weights.*mask_moc;
                filtered_img(i, j) = sum(weights.*matrix_of_consideration, 'all')/sum(weights, 'all');
            end
        end
end