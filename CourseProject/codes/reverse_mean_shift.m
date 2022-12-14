clear; clc; close all;

im = double(imread ('lena_256.bmp'));
figure(1); imshow(im/255);
[H,W] = size(im);
% im1 = imgaussfilt(im,2);
im1 = space_varying_gaussian(im, 1, 3);
% im1 = im1 + randn(H,W)*3;
figure(2); imshow(im1/255);

display("PSNR of blurred/noisy image:")
disp(compute_psnr(im,im1));

im2 = restore_image(im, im1, 20, 100, 2, 10, 0.01, 0.1);
figure(3); imshow(im2/255);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function im2 = restore_image(im,im1, sig1, sig2, sig3, sig4, step_frwd, step_rev)
% Image restoration function
% Inputs:
%       - im: original image
%       - im1: blurr/noisy image
%       - sig1: spacial standard deviation for reverse mean shift
%       - sig2: intensity standard deviation for reverse mean shift
%       - sig3: spacial standard deviation for forward mean shift
%       - sig4: intensity standard deviation for forward mean shift
%       - step_frwd: step for forward mean shift
%       - step_rev: step for reverse mean shift
% Returns:
%       - im2: deblurred image using reverse mean shift
    psnr_old = 0;
    psnr_new = 0;
    delta_psnr = 0;
    im2 = im1;
    for i = 1:50
        im2 = reverse_mean_shift_step(im2, sig1, sig2, step_rev);
        im2 = forward_mean_shift_step(im2, sig3, sig4, step_frwd);
        psnr_old = psnr_new;
        psnr_new = compute_psnr(im,im2);
        delta_psnr = psnr_new - psnr_old;
        if delta_psnr < 0
            break;
        end
        disp([i, compute_psnr(im,im2), compute_ncc(im-im1,im2)]);
    end

end


function im2 = reverse_mean_shift_step(im, sig1, sig2, step)
% Implementation of single step of reverse mean shift
% Inputs:
%       - im: blurr/noisy image
%       - sig1: spacial standard deviation for reverse mean shift
%       - sig2: intensity standard deviation for reverse mean shift
%       - step: step for reverse mean shift
% Returns:
%       - im2: deblurred image using reverse mean shift for single step
    [H,W] = size(im);
    P = 3; % radius of window to compute mean, so window will be 7x7
    im2 = im; im2(:,:) = 0;

    for i=1:H
        for j=1:W
            curr = [i j squeeze(im(i,j))']; % a point is [x,y,I]
            currx = floor(curr(2)); 
            curry = floor(curr(1));            
            [X,Y] = meshgrid(max(currx-P,1):min(currx+P,W),max(curry-P,1):min(curry+P,H));
            Z = im(max(curry-P,1):min(curry+P,H),max(currx-P,1):min(currx+P,W));
            weights = exp(-((X-j).^2+(Y-i).^2)/(2*sig1*sig1)-((Z-im(i,j)).^2)/(2*sig2*sig2));
            curr(3) = curr(3) - step * (sum(weights(:).*Z(:))/sum(weights(:))-curr(3));
            im2(i,j) = curr(3);  % copy color values into "filtered image"
        end
    end
    im2(im2 < 0)= 0;
    im2(im2 > 255) = 255;
    
end


function im2 = forward_mean_shift_step(im, sig1, sig2, step)
% Implementation of single step of reverse mean shift
% Inputs:
%       - im: blurr/noisy image
%       - sig1: spacial standard deviation for forward mean shift
%       - sig2: intensity standard deviation for forward mean shift
%       - step: step for forward mean shift
% Returns:
%       - im2: denoised image using forward mean shift for single step   
    [H,W] = size(im);
    P = 3; % radius of window to compute mean, so window will be 7x7
    im2 = im; im2(:,:) = 0;

    for i=1:H
        for j=1:W
            curr = [i j squeeze(im(i,j))']; % a point is [x,y,I]
            currx = floor(curr(2)); 
            curry = floor(curr(1));            
            [X,Y] = meshgrid(max(currx-P,1):min(currx+P,W),max(curry-P,1):min(curry+P,H));
            Z = im(max(curry-P,1):min(curry+P,H),max(currx-P,1):min(currx+P,W));
            weights = exp(-((X-j).^2+(Y-i).^2)/(2*sig1*sig1)-((Z-im(i,j)).^2)/(2*sig2*sig2));
            curr(3) = curr(3) + step * (sum(weights(:).*Z(:))/sum(weights(:))-curr(3));
            im2(i,j) = curr(3);  % copy color values into "filtered image"      
        end
    end
    im2(im2 < 0)= 0;
    im2(im2 > 255) = 255;
    
end


function psnr = compute_psnr(img1, img2)
% Computes Peak SNR between original and deblurred image
% Inputs:
%       - Original image: img1
%       - Deblurred image: img2
% Returns:
%       - Peak SNR: psnr

    max_I = max(img2, [], 'all');
    img_shape = size(img1);
    mse = sum((img1-img2).^2, 'all')/(img_shape(1)*img_shape(2));
    psnr = 20*log10(max_I/(mse^0.5));
end


function ncc = compute_ncc(im1, im2)
% Computed ncc between two images
% Inputs:
%       - im1: image before the reverse mean shift iteration
%       - im2: image after reverse mean shift iteration
% Returns:
%       - ncc: correlation between im1 and im2
    m1 = mean(im1,'all');
    m2 = mean(im2,'all');
    ncc = abs(sum((im1-m1).*(im2-m2),'all')/(sqrt(sum((im1-m1).^2,'all'))*sqrt(sum((im2-m2).^2,'all'))));
end


function blurred_img = space_varying_gaussian(img, sigma_min, sigma_max)
% Blurs an image with a gaussian filter
% Inputs:
%       - Image: img
%       - sigma_min: minimum standard deviation of the gaussian
%       - sigma_max: maximum standard deviation of the gaussian
% Returns:
%       - blurred_img: space varying blurred image for our analysis
    [h,w] = size(img);
    sigmas = linspace(sigma_min, sigma_max, w/4);
    blurred_img = zeros(size(img));
    j = 1;
    for i=1:4:w
        temp = imgaussfilt(img, sigmas(j));
        blurred_img(:,i:i+3) = temp(:,i:i+3);
        j = j+1;
    end
end