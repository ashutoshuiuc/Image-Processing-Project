%% MyMainScript

tic;

%% Your code here

clear; clc;

%% Load the images

% read in the images barbara256.png and stream.png as double arrays
I1 = double(imread('barbara256.png'));
I2 = double(imread('stream.png'));
I2 = I2(1:256,1:256);

% Add gaussian noise with 20 standard deviation to both images
sigma = 20;
im1 = I1 + randn(size(I1))*sigma;
im2 = I2 + randn(size(I2))*sigma;

figure(1); imshow((im1-min(im1(:)))/(max(im1(:))-min(im1(:))));
figure(2); imshow((im2-min(im2(:)))/(max(im2(:))-min(im2(:))));

rmse_im1 = norm(im1-I1,"fro")/norm(I1,"fro");
rmse_im2 = norm(im2-I2,"fro")/norm(I2,"fro");

%% (a)

im1_a = myPCADenoising1(im1);
im2_a = myPCADenoising1(im2);
rmse_im1_a = norm(im1_a-I1,"fro")/norm(I1,"fro");
rmse_im2_a = norm(im2_a-I2,"fro")/norm(I2,"fro");

figure(3); imshow((im1_a-min(im1_a(:)))/(max(im1_a(:))-min(im1_a(:))));
figure(4); imshow((im2_a-min(im2_a(:)))/(max(im2_a(:))-min(im2_a(:))));

%% (b)

im1_b = myPCADenoising2(im1);
im2_b = myPCADenoising2(im2);
rmse_im1_b = norm(im1_b-I1,"fro")/norm(I1,"fro");
rmse_im2_b = norm(im2_b-I2,"fro")/norm(I2,"fro");

figure(5); imshow((im1_b-min(im1_b(:)))/(max(im1_b(:))-min(im1_b(:))));
figure(6); imshow((im2_b-min(im2_b(:)))/(max(im2_b(:))-min(im2_b(:))));

%% (c)

im1_c = myBilateralFilter(im1, 15, 20);
im2_c = myBilateralFilter(im2, 15, 20);
rmse_im1_c = norm(im1_c-I1,"fro")/norm(I1,"fro");
rmse_im2_c = norm(im2_c-I2,"fro")/norm(I2,"fro");

figure(7); imshow((im1_c-min(im1_c(:)))/(max(im1_c(:))-min(im1_c(:))));
figure(8); imshow((im2_c-min(im2_c(:)))/(max(im2_c(:))-min(im2_c(:))));

toc;

%% Function definitions 

function final_output = myPCADenoising1(noisy_input)
% Does PCA denoising on the input image
% Inputs:
%       - Image: img
% Returns:
%       - Image filtered using PCA denoising: filtered_img
    
    [h, w] = size(noisy_input);
    sigma = 20;
    p = 7;
    factor = zeros(size(noisy_input));
    patches = zeros(p^2, (h-p+1)*(w-p+1));
    output = zeros(size(noisy_input));
    
    for i = 1:h-p+1
        for j = 1:w-p+1
            patch = noisy_input(i:i+p-1, j:j+p-1);
            patches(:, (h-p+1)*(i-1)+j) = patch(:);
            factor(i:i+p-1, j:j+p-1) = factor(i:i+p-1, j:j+p-1) + 1;
        end
    end
    
    [V,D] = eig(patches*transpose(patches), 'vector');

    alpha = transpose(V)*patches;
    alpha_j_squared = max(0, (sum(alpha.^2,2)/size(patches,2))-sigma^2);

    alpha_denoised = alpha./(1+(sigma^2)./alpha_j_squared);
    alpha_denoised = V*alpha_denoised;
    
    for i = 1:h-p+1
        for j = 1:w-p+1
            output(i:i+p-1, j:j+p-1) = output(i:i+p-1, j:j+p-1) + reshape(alpha_denoised(:,(h-p+1)*(i-1)+j),[p,p]);
        end
    end

    final_output = output./factor;

end

function final_output = myPCADenoising2(noisy_input)
% Does PCA denoising on the input image
% Inputs:
%       - Image: img
% Returns:
%       - Image filtered using PCA denoising: filtered_img
    
    [h, w] = size(noisy_input);
    sigma = 20;
    p = 7;
    factor = zeros(size(noisy_input));
    output = zeros(size(noisy_input));
    
    for i = 1:h-p+1
        
        for j = 1:w-p+1
            patch = noisy_input(i:i+p-1, j:j+p-1);
            patch = patch(:);
            h_min = max(i-15,1);
            w_min = max(j-15,1);
            h_max = min(i+15-p+1,h-p+1);
            w_max = min(j+15-p+1,w-p+1);
            ngbd = noisy_input(h_min:h_max, w_min:w_max);
            ngbd_patches = zeros(p^2, (h_max-h_min+1)*(w_max-w_min+1));
            
            for k = h_min:h_max
                for l = w_min:w_max
                    ngbd_patch = noisy_input(k:k+p-1, l:l+p-1);
                    ngbd_patches(:, (w_max-w_min+1)*(k-h_min)+(l-w_min+1)) = ngbd_patch(:);
                end
            end
            
            idx = knnsearch(transpose(ngbd_patches), transpose(patch), "K", 200);
            Q = ngbd_patches(:,idx);
            [V,D] = eig(Q*transpose(Q), 'vector');
            
            alpha = transpose(V)*Q;
            alpha_j_squared = max(0,(sum(alpha.^2,2)/size(Q,2))-sigma^2);
            
            alpha_denoised = alpha./(1+(sigma^2)./alpha_j_squared);
            alpha_denoised = V*alpha_denoised(:,1);
            
            output(i:i+p-1, j:j+p-1) = output(i:i+p-1, j:j+p-1) + reshape(alpha_denoised(:),[p,p]);
            factor(i:i+p-1, j:j+p-1) = factor(i:i+p-1, j:j+p-1) + 1;
            
        end
    end
    
    final_output = output./factor;
    
end

function filtered_img = myBilateralFilter(img, sigma_s, sigma_r)
% Does bilateral filtering on the input image using the given sigma_s and
% sigma_r values
% Inputs:
%       - Image: img
%       - Standard deviation for space based weights: sigma_s
%       - Standard deviation for intensity based weights: sigma_r
% Returns:
%       - Image filtered using bilateral filter: filtered_img
    window_size = ceil(6*sigma_s);
    if rem(window_size, 2) == 0
        window_size = window_size + 1;
    end
    
    filter_matrix1 = zeros([window_size window_size]);
    for i = 1:window_size
        for j = 1:window_size
            vec1 = [i j];
            vec2 = [fix(window_size/2) fix(window_size/2)];
            filter_matrix1(i, j) = filter_matrix1(i, j) + norm(vec1 - vec2);
        end
    end
    
    space_weights = exp(-(filter_matrix1.*filter_matrix1)/(2*sigma_s*sigma_s))/(sigma_s*(2*pi)^0.5);
    
    pad_size = fix(window_size/2);
    img_padded = padarray(img,[pad_size pad_size],0,'both');
    filtered_img = zeros(size(img));
    h = size(img, 1);
    w = size(img, 2);
        for i = 1:h
            for j = 1:w
                matrix_of_consideration = img_padded(i:i+window_size-1, j:j+window_size-1);
                filter_matrix2 = abs(matrix_of_consideration - matrix_of_consideration(pad_size+1, pad_size+1));
                intensity_weights = exp(-(filter_matrix2.*filter_matrix2)/(2*sigma_r*sigma_r))/(sigma_r*(2*pi)^0.5);
                weights = space_weights.*intensity_weights;
                filtered_img(i, j) = sum(weights.*matrix_of_consideration, 'all')/sum(weights, 'all');
            end
        end
end