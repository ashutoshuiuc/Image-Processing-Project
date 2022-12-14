%% MyMainScript

tic;
%% Your code here

clc; clear;

%% QUSETION 4

% Read in the images T1.jpg and T2.jpg as double arrays
j1 = imread('T1.jpg');
j2 = imread('T2.jpg');
j1 = double(j1)+1;
j2 = double(j2)+1;

%% PART (a)

% Rotate j2 by 28.5 degree anti-clockwise using bilinear interpolation
% imrotate() by default assigns 0 value to unoccupied pixels
% Crop the rotated image to the same size as the original one
j3 = double(imrotate(j2, 28.5, 'bilinear', 'crop'));

% Plot the rotated image for visualization
figure(1); imshow(j3/255);

%% PART (b)

x = [-45:45]; % list of angles in range -45 to 45 degrees in steps of 1 degree
output_ncc = zeros(1, 91); % output ncc for all rotations initialized to 0
output_je = zeros(1, 91); % output je for all rotations initialized to 0
output_qmi = zeros(1, 91); % output qmi for all rotations initialized to 0

for i = -45:45
    j4 = double(imrotate(j3, i, 'bilinear', 'crop'));
    output_ncc(1, 46+i) = calculate_ncc(j1, j4);
    output_je(1, 46+i) = calculate_je(j1, j4);
    output_qmi(1, 46+i) = calculate_qmi(j1, j4);
end

%% PART (c),(d)
figure(2)
plot(x, output_ncc); 
xlabel('Angle of Rotation (degrees)'); ylabel('Normalized Cross-Correlation'); title('Normalized Cross-Correlation');
grid on;
grid minor;

figure(3);
plot(x, output_je);
xlabel('Angle of Rotation (degrees)'); ylabel('Joint Entropy'); title('Joint Entropy');
grid on;
grid minor;

figure(4);
plot(x, output_qmi);
xlabel('Angle of Rotation (degrees)'); ylabel('Quadratic Mutual Information'); title('Quadratic Mutual Information');
grid on;
grid minor;

%% PART (e)

angle_je = -29; % optimal rotation angle obtained using joint entropy
j4 = double(imrotate(j3, angle_je, 'bilinear', 'crop'));
joint_hist_j1_j4 = calculate_joint_hist(j1, j4, 10);
figure(5);
imagesc(joint_hist_j1_j4); colorbar;
xlabel('Intensity Value Bin for Image J1'); ylabel('Intensity Value Bin for Image J4'); title('Joint Histogram');

toc;
%% Function definitions

function ncc = calculate_ncc(img1, img2)
% NCC Computes Normalized Cross-Correlation of two images.
% Inputs:
%       - Image-1: img1 
%       - Image-2: img2
% Returns:
%       - Normalized Cross-Correlation of img1 and img2: ncc
    mask_zeros = (img2>0);
    img_1 = img1-1;
    img_2 = img2-1;
    img_1 = img_1.*mask_zeros;
    img_2 = img_2.*mask_zeros;
    mean_img1 = sum(img_1, "all")/sum(mask_zeros, "all");
    mean_img2 = sum(img_2, "all")/sum(mask_zeros, "all");
    num = sum(((img_1-mean_img1).*(img_2-mean_img2).*mask_zeros), 'all');
    den = (sum(((img_1-mean_img1).*mask_zeros).^2, 'all') * sum(((img_2-mean_img2).*mask_zeros).^2, 'all')).^0.5;
    ncc = abs(num/den);
end

function joint_hist = calculate_joint_hist(img1, img2, bin_width)
% JOINT_HIST Computes Joint Histogram of two images.
% Inputs:
%       - Image-1: img1 
%       - Image-2: img2
%       - Size of the bin: bin_width
% Returns:
%       - Joint Histogram of img1 and img2 with bin size bin_width: joint_hist
    mask_zeros = (img2>0);
    joint_hist = zeros(fix(255/bin_width)+1); % additional bin for leftover pixels
    h = size(img1,1);
    w = size(img1,2);
    num_masked = 0;
    for i = 1:h
        for j = 1:w
            if mask_zeros(i, j) == 1
                x = fix((img2(i,j)-1)/bin_width)+1;
                y = fix((img1(i,j)-1)/bin_width)+1;
                joint_hist(x,y) = joint_hist(x,y) + 1;
            else
                num_masked = num_masked+1;
            end
        end
    end
    joint_hist = joint_hist/(sum(mask_zeros, "all"));
end

function marginal_hist = calculate_marginal_hist(joint_hist, index)
% MARGINAL_HIST Computes Marginal Histogram.
% Inputs:
%       - Joint Histogram: joint_hist 
%       - Image Number: index
% Returns:
%       - Marginal Histogram: marginal_hist
    marginal_hist = sum(joint_hist, index);
end

function je = calculate_je(img1, img2)
% JE Computes Joint Entropy.
% Inputs:
%       - Image-1: img1 
%       - Image-2: img2
% Returns:
%       - Joint Entropy: je
    joint_hist = calculate_joint_hist(img1, img2, 10);
    joint_hist(joint_hist == 0) = 1; % set joint_hist to be 1 where it is 
    % coming out to be 0 to avoid 0 inside log
    je = -1*sum(joint_hist.*(log(joint_hist)/log(2)), 'all');
end

function qmi = calculate_qmi(img1, img2)
% QMI Computes Quadratic Mutual Information.
% Inputs:
%       - Image-1: img1 
%       - Image-2: img2
% Returns:
%       - Quadratic Mutual Information: qmi
    joint_hist = calculate_joint_hist(img1, img2, 10);
    marginal_hist_img1 = calculate_marginal_hist(joint_hist, 1);
    marginal_hist_img2 = calculate_marginal_hist(joint_hist, 2);
    qmi = sum((joint_hist - marginal_hist_img2*marginal_hist_img1).^2, 'all');
end
