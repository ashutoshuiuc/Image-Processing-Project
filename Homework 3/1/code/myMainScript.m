%% MyMainScript

tic;
%% Your code here

clc; clear;

% read in the images barbara256.png and kodak24.png as double arrays
I1 = double(imread('barbara256.png'))+1;
I2 = double(imread('kodak24.png'))+1;

% Add gaussian noise with zero mean and 5 standard deviation to both
% images

I1 = I1 + normrnd(0, 5, size(I1));
I2 = I2 + normrnd(0, 5, size(I2));

% % Uncomment below to add gaussian noise with zero mean and 10 standard 
% % deviation to both images
% 
% I1 = I1 + normrnd(0, 10, size(I1));
% I2 = I2 + normrnd(0, 10, size(I2));

figure(1); imshow(I1/255); colormap(gray);

filt_img = meanshiftfilter(I1, 2, 2, 1);
figure(2); imshow(filt_img/255); colormap(gray);

filt_img = meanshiftfilter(I1, 0.1, 0.1, 1);
figure(3); imshow(filt_img/255); colormap(gray);

filt_img = meanshiftfilter(I1, 3, 15, 1);
figure(4); imshow(filt_img/255); colormap(gray);

figure(5); imshow(I2/255); colormap(gray);

filt_img = meanshiftfilter(I2, 2, 2, 1);
figure(6); imshow(filt_img/255); colormap(gray);

filt_img = meanshiftfilter(I2, 0.1, 0.1, 1);
figure(7); imshow(filt_img/255); colormap(gray);

filt_img = meanshiftfilter(I2, 3, 15, 1);
figure(8); imshow(filt_img/255); colormap(gray);

toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function filtered_img = meanshiftfilter(img, sigma_s, sigma_r, epsilon)
% Does smoothing based on mean shift algorithm on the input image using the
% given sigma_s and sigma_r values, until convergence given by epsilon
% Inputs:
%       - Image: img
%       - Standard deviation for space based weights: sigma_s
%       - Standard deviation for intensity based weights: sigma_r
%       - Convergence parameter: epsilon
% Returns:
%       - Image filtered using mean shift filter: filtered_img
    window_size = ceil(6*sigma_s);
    if rem(window_size, 2) == 0
        window_size = window_size + 1;
    end

    if window_size < 13
        window_size = 13;
    end

    display(window_size)
    
    filter_matrix1 = zeros([window_size, window_size]);
    for i = 1:window_size
        for j = 1:window_size
            vec1 = [i, j];
            vec2 = [fix(window_size/2)+1, fix(window_size/2)+1];
            filter_matrix1(i, j) = norm(vec1 - vec2)^2/(2*sigma_s*sigma_s);
        end
    end
    space_weights = double(exp(-(filter_matrix1)));
    
    pad_size = fix(window_size/2);
    img_padded = padarray(img,[pad_size pad_size],0,'both');
    mask = (img_padded>1);
    img_padded = img_padded -1;
    filtered_img = zeros(size(img));
    h = size(img, 1);
    w = size(img, 2);

    for i = 1:h
        for j = 1:w
            display([i, j]);
            curr_vec = [i, j, img(i, j)];
            errors = [];
            err = inf;
            diff = inf;
            errors = cat(1, errors, [err]);
            while diff > epsilon
                box1 = zeros([window_size, window_size]);
                for k = 1:window_size
                    for l = 1:window_size
                        box1(k, l) = k+curr_vec(1)-1-pad_size;
                    end
                end
                
                box2 = zeros([window_size, window_size]);
                for k = 1:window_size
                    for l = 1:window_size
                        box2(k, l) = l+curr_vec(2)-1-pad_size;
                    end
                end
                box3 = img_padded(curr_vec(1):curr_vec(1)+window_size-1, curr_vec(2):curr_vec(2)+window_size-1);
                mask_to_be_considered = mask(curr_vec(1):curr_vec(1)+window_size-1, curr_vec(2):curr_vec(2)+window_size-1);
                filter_matrix2 = abs(box3 - box3(pad_size+1, pad_size+1));
                intensity_weights = exp(-(filter_matrix2.*filter_matrix2)/(2*sigma_r*sigma_r));
                weights = space_weights.*intensity_weights.*mask_to_be_considered;
                num1 = sum(box1.*weights, 'all')/sum(weights, 'all');
                num2 = sum(box2.*weights, 'all')/sum(weights, 'all');
                num3 = sum(box3.*weights, 'all')/sum(weights, 'all');
                new_vec = [min(h, max(1, num1)), min(w, max(1, num2)), num3];
                err = norm(curr_vec - new_vec);
                diff = - err + errors(end);
                errors = cat(1, errors, [err]);
                curr_vec = [ceil(new_vec(1)), ceil(new_vec(2)), new_vec(3)];
            end
            filtered_img(i, j) = img(curr_vec(1), curr_vec(2));
        end
    end
end
