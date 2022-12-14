%% MyMainScript

tic;
%% Your code here

clear; clc;

% read in the image barbara256.png as double array
img = double(imread('barbara256.png'));

figure(1);
imshow(img/255); colormap(gray);

dft_center = fftshift(fft2(img));
abs_dft = log(abs(dft_center)+1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% For changing the parameter values of the ideal and gaussian low pass filters between 40 and 80, 
% change the last function argument in  lines 25 and 46.

figure(2);
imshow(abs_dft,[min(abs_dft(:)) max(abs_dft(:))]); colormap (jet); colorbar;

[ILPF_freq_response, img_ILPF_freq] = ILPF_freq(img, 40);
abs_ILPF_freq_response = log(abs(ILPF_freq_response)+1);
abs_ILPF_freq = log(abs(img_ILPF_freq)+1);

figure(3);
imshow(abs_ILPF_freq_response,[min(abs_ILPF_freq_response(:)) max(abs_ILPF_freq_response(:))]); colormap (jet); colorbar;

figure(4);
imshow(abs_ILPF_freq,[min(abs_ILPF_freq(:)) max(abs_ILPF_freq(:))]); colormap (jet); colorbar;

filtered_img_padded = ifft2(img_ILPF_freq);
h = size(filtered_img_padded,1);
w = size(filtered_img_padded,2);
filtered_img = filtered_img_padded(fix(h/4):fix(h/2)+fix(h/4), fix(w/4):fix(w/2)+fix(w/4));

figure(5);
imshow(abs(filtered_img)/255); colormap(gray);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[GLPF_freq_response, img_GLPF_freq] = GLPF_freq(img, 40);
abs_GLPF_freq_response = log(abs(GLPF_freq_response)+1);
abs_GLPF_freq = log(abs(img_GLPF_freq)+1);

figure(6);
imshow(abs_GLPF_freq_response,[min(abs_GLPF_freq_response(:)) max(abs_GLPF_freq_response(:))]); colormap (jet); colorbar;

figure(7);
imshow(abs_GLPF_freq,[min(abs_GLPF_freq(:)) max(abs_GLPF_freq(:))]); colormap (jet); colorbar;

filtered_img_padded = ifft2(img_GLPF_freq);
h = size(filtered_img_padded,1);
w = size(filtered_img_padded,2);
filtered_img = filtered_img_padded(fix(h/4):fix(h/2)+fix(h/4), fix(w/4):fix(w/2)+fix(w/4));

figure(8);
imshow(abs(filtered_img)/255); colormap(gray);

toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [filter, filtered_img_freq] = ILPF_freq(img, d)
% Filters the image by cutting off frequencies above d using an ideal low
% pass filter
% Inputs:
%       - Image: img
%       - cut off frequency: d
% Returns:
%       - Image filtered using ideal low pass filter: filtered_img
    h = size(img,1);
    w = size(img,2);
    filter = zeros([2*h 2*w]);
    img_padded = padarray(img, [fix(w/2) fix(h/2)],0,'both');
    dft_center = fftshift(fft2(img_padded));
    for i = 1:2*h
        for j = 1:2*w
            if ((i-h)^2 + (j-w)^2) <= d^2
                filter(i,j) = 1;
            end
        end
    end
    filtered_img_freq = filter.*dft_center;
end

function [filter, filtered_img_freq] = GLPF_freq(img, sigma)
% Filters the image using a gaussian filter for low pass filter
% Inputs:
%       - Image: img
%       - Standard deviation of the gaussian: sigma
% Returns:
%       - Image filtered using gaussian low pass filter: filtered_img
    h = size(img,1);
    w = size(img,2);
    filter = zeros([2*h 2*w]);
    img_padded = padarray(img, [fix(w/2) fix(h/2)],0,'both');
    dft_center = fftshift(fft2(img_padded));
    for i = 1:2*h
        for j = 1:2*w
            filter(i,j) = exp(-((i-h)^2+(j-w)^2)/(2*sigma*sigma));
        end
    end
    filtered_img_freq = filter.*dft_center; 
end
