clear; clc;

org_img = double(imread("barbara256.png"));
% org_img = org_img(200:400, 200:400);

figure(1);
imshow(org_img/255); colormap(gray);

% img = gaussian_blurring(org_img, 2, 50);
img = space_varying_gaussian(org_img, 1, 3);
figure(2);
imshow(img/255); colormap(gray);
org_img = padarray(org_img, [1 1], 'replicate', 'post'); 
org_img = padarray(org_img, [1 1], 'replicate', 'pre');

img = padarray(img, [1 1], 'replicate', 'post'); % did appropriate padding 
% to enforce neumann boundary conditions when computing derivatives
img = padarray(img, [1 1], 'replicate', 'pre');

% The following 4 commented lines were used during grid search of
% parameters
% cs = linspace(0.1, 1, 10);
% betas = linspace(0.001, 0.1, 10);
% dts = linspace(0.01, 10, 10);
% psnrs = [];

deblurred_img = rev_heat_eqn_deblur(org_img, img, 1.2, 0.06, 0.1, 10, 1);
figure(3);
imshow(deblurred_img/255); colormap(gray);

display("PSNR of blurred image:")
display(compute_psnr(org_img, img));

% Grid search of parameters
% for c = cs
%     for beta = betas
%         for dt = dts
%             deblurred_img = rev_heat_eqn_deblur(img, c, beta, dt, 100, 0.3);
%             psnr = compute_psnr(org_img, deblurred_img);
%             display([c, beta, dt, psnr]);
%             psnrs = [psnrs, psnr];
%         end
%     end
% end

display("PSNR of deblurred image:")
display(compute_psnr(org_img, deblurred_img));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function psnr = compute_psnr(img1, img2)
% Computes Peak SNR between original and deblurred image
% Inputs:
%       - Original image: img1
%       - Deblurred image: img2
% Returns:
%       - Peak SNR: psnr

    img_shape = size(img1);
    mse = sum((img1-img2).^2, 'all')/(img_shape(1)*img_shape(2));
    psnr = 20*log10(255/(mse^0.5));
end


function curvature = compute_curvature(img)
% Computes curvature of the image
% Inputs:
%       - Image: img
% Returns:
%       - curvature: curvature of the image
%   Compute all the relevant image derivatives which will be needed for our
%   analysis:

    [u_x , u_y] = imgradientxy(img, 'intermediate'); % https://in.mathworks.com/help/images/ref/imgradientxy.html
    [u_xx , u_yx] = gradient(u_x);
    [u_xy , u_yy] = gradient(u_y);
    
    curvature = (u_xx.*u_x.*u_x - 2*u_xy.*u_x.*u_y + u_yy.*u_y.*u_y)./((u_x.*u_x + u_y.*u_y).^1.5); % an implementation of equation 14 of our paper
end


function deblurred_img = rev_heat_eqn_deblur(org_img, img, c, beta, dt, n_iter, curv_threshold)
% Deblurs the image using reverse heat equation
% Inputs:
%       - Image: img
%       - c: diffusion coefficient
%       - beta: fraction of forward component of diffusion along the normal
%       - dt: time interval over which we want to update
%       - n_iter: total number of iterations for deblurring
%       - curv_threshold: curvature change threshold
% Returns:
%       - deblurred_img: deblurred image using reverse heat equation
    u_new = img;
    mask = ones(size(img));
    curvature_old = compute_curvature(u_new);
    curvature_new = curvature_old;
    img_shape = size(img);

    for i = 1:n_iter
        u_t = rev_heat_eqn_single_iter(u_new, c, beta); % compute the time derivative
        u_new = u_new + mask.*u_t*dt; % update the image
        u_new(u_new>255) = 255;
        u_new(u_new<0) = 0;
        curvature_old = curvature_new;
        curvature_new = compute_curvature(u_new);
        mask = (abs(curvature_new - curvature_old) < curv_threshold);
        psnr = compute_psnr(org_img, u_new);
        display([i, psnr])
        mask2 = (abs(curvature_new - curvature_old) > curv_threshold);
        display(sum(mask2, 'all'));
        if sum(mask2, 'all') > 1*img_shape(1)*img_shape(2) % max(abs(curvature_new - curvature_old), [], 'all') < curv_threshold
           display(i);
           break;
        end
    end
    deblurred_img = u_new;
end


function one_iter_deblur_t = rev_heat_eqn_single_iter(img, c, beta)
% Computes the reverse time derivative of the image, an implementation
% of equation 9 of our paper
% Inputs:
%       - Image: img
%       - c: diffusion coefficient
%       - beta: fraction of forward component of diffusion along the normal
% Returns:
%       - one_iter_deblur_t: reverse time derivative of the image
%   Compute all the relevant image derivatives which will be needed for our
%   analysis:

    [u_x , u_y] = imgradientxy(img, "intermediate"); % https://in.mathworks.com/help/images/ref/imgradientxy.html
    [u_xx , u_yx] = gradient(u_x);
    [u_xy , u_yy] = gradient(u_y);
    
    u_nn = (u_xx.*u_x.*u_x + 2*u_xy.*u_x.*u_y + u_yy.*u_y.*u_y)./(u_x.*u_x + u_y.*u_y + 1); % an implementation of equation 7 of our paper
    u_zz = (u_xx.*u_x.*u_x - 2*u_xy.*u_x.*u_y + u_yy.*u_y.*u_y)./(u_x.*u_x + u_y.*u_y + 1); % an implementation of equation 8 of our paper 

    u_laplacian = 4*del2(img); % u_nn + u_zz; % an implementation of equation 6 of our paper 
    u_t = -c*u_laplacian + beta*u_nn; % an implementation of equation 9 of our paper 

    one_iter_deblur_t = u_t;
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


function blurred_img = gaussian_blurring(img, sigma, kernel_size)
% Blurs an image with a gaussian filter
% Inputs:
%       - Image: img
%       - sigma: standard deviation of the gaussian
%       - kernel_size: kernel size of the gaussian filter for blurring
% Returns:
%       - blurred_img: blurred image for our analysis
    kernel = fspecial('gaussian', [kernel_size kernel_size], sigma);
    blurred_img = imfilter(img, kernel);
end