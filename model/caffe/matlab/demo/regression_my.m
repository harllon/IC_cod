% function output = regression_my(input_data)
clear all

% Add caffe/matlab to you Matlab search PATH to use matcaffe
if exist('../+caffe', 'dir')
  addpath('..');
else
  error('Please run this demo from caffe/matlab/demo');
end
caffe.reset_all();

caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

% Initialize the network using BVLC CaffeNet for image classification
% Weights (parameter) file needs to be downloaded from Model Zoo.
model_dir = '../../models/viewInterp/';
net_model = [model_dir 'deploy_direct.prototxt'];
net_weights = [model_dir 'model/direct_iter_60000.caffemodel'];
phase = 'test'; % run with phase test (so that dropout isn't applied)
if ~exist(net_weights, 'file')
  error('Please download CaffeNet from Model Zoo before you run this demo');
end

addpath '/media/C/Users/tcwang/Desktop/Code/4_LearningCode_MATLAB/Functions/'

InitParam(); global param;
skipNum = param.skipNum;
fullAngHeight = param.fullAngHeight;
fullAngWidth = param.fullAngWidth;

datasetFolder = '/media/C/Users/tcwang/Desktop/Code/1_Datasets/';
trainingFolder = [datasetFolder, '/Train/'];
testingFolder = [datasetFolder, '/Test/'];
[sceneNames, scenePaths, numScenes] = GetFolderContent(testingFolder);
tmpViewX = [1, 1, 2, 2];
tmpViewY = [1, 2, 1, 2];
numViews = length(tmpViewX);
% si = 1;
for si = 1:numScenes
% features = [];
% mkdir(sceneNames{si}(1:end-3))
[sparseLF, fullLF] = ReadInputLightField(scenePaths{si}, skipNum);
% for vi = 1 : numViews
%         %%% extract features
%         ahi = tmpViewY(vi);
%         awi = tmpViewX(vi);
%         
%         curFolder = [datasetFolder, '/Features-Lytro/', sceneNames{si}(1:end-3)];
%         curFileName = sprintf('%s/feature-%02d-%02d.h5', curFolder, ahi, awi);
% %         features = cat(3, features, h5read(curFileName, '/FT'));
%         feature = h5read(curFileName, '/FT'); feature(:,:,17:end) = sparseLF(:, :, :, ahi, awi);
%         features = cat(3, features, feature);
% end
[h, w, ~] = size(sparseLF);
features = permute(sparseLF, [1 2 4 5 3]);
features = reshape(features, h, w, []);

patch_size = 200;
offset = 16;
d_size = patch_size-offset;

features = padarray(features, [offset/2 offset/2], 'replicate', 'pre');
features = padarray(features, [patch_size-offset/2 patch_size-offset/2], 'replicate', 'post');

tic;
net = caffe.Net(net_model, net_weights, phase);
for refViewY = 1:8    
for refViewX = 1:8
% refViewX = 8;
dy = ((refViewY-1)/skipNum+1) - 1;%tmpViewY(vi);
dx = ((refViewX-1)/skipNum+1) - 1;%tmpViewX(vi);
gt = fullLF(:,:,:,refViewY,refViewX);

testData = [];
for m = 1 : ceil(w/(patch_size-offset))
    for n = 1 : ceil(h/(patch_size-offset))
        x_idx = (m-1)*(patch_size-offset) + (1:patch_size);
        y_idx = (n-1)*(patch_size-offset) + (1:patch_size);
        testData = cat(4, testData, features(y_idx, x_idx, :));
    end
end

test_output = zeros(h, w, 3);
% net = caffe.Net(net_model, net_weights, phase);
for bb = 1 : size(testData, 4)
    input_data{1} = testData(:,:,:,bb);
    input_data{2} = reshape(dx, [1 1 1 length(dx)]);
    input_data{3} = reshape(dy, [1 1 1 length(dy)]);     
    output_data = net.forward(input_data);
    output_data = squeeze(output_data{1});
    output_data = output_data(1+offset/2:end-offset/2, 1+offset/2:end-offset/2, :);
    i = mod((bb-1), ceil(h/(patch_size-offset)))*d_size + (1:d_size);
    j = floor((bb-1)/ceil(h/(patch_size-offset)))*d_size + (1:d_size);
    test_output(i, j, :) = output_data;           
end
% caffe.reset_all();
test_output = test_output(1:h, 1:w, :, :);
imwrite(test_output, [sceneNames{si}(1:end-3) '/direct_' num2str(refViewY-1) '_' num2str(refViewX-1) '.png'])
% figure('Position', [100, 100, 1000, 1000]);imshow(gt)
% figure('Position', [100, 100, 1000, 1000]);imshow(test_output)
end
end
caffe.reset_all();
toc
end


% input_data{1} = features(1:patch_size,1:patch_size,:,:);
% input_data{2} = reshape(dx, [1 1 1 length(dx)]);
% input_data{3} = reshape(dy, [1 1 1 length(dy)]); 
% 
% % Initialize a network
% net = caffe.Net(net_model, net_weights, phase);
% 
% tic;
% output = net.forward(input_data);
% output = output{1};
% % output = permute(output, [2 1 3 4]);
% toc;
% figure('Position', [100, 100, 1000, 1000]);imshow(gt,'InitialMagnification', 300)
% figure('Position', [100, 100, 1000, 1000]);imshow(output,'InitialMagnification', 300)
% % imwrite(output, ['output' num2str(si, '%02d') '.png'])
% % imwrite(gt, ['gt' num2str(si, '%02d') '.png'])
% caffe.reset_all();
% % end



% % fileName = '/media/C/Users/tcwang/Desktop/Code/2_TrainingData/ProcessedTrainingStage1-Lytro/1/IMG_1468/training.h5';
% fe = zeros(16,1);
% ff = 1;
% for f = 1510%:1525    
% fileName = sprintf('/media/C/Users/tcwang/Desktop/Code/2_TrainingData/ProcessedValidationStage1-Lytro/1/IMG_%d/training.h5', f);
% for i = 1%:12
% feat = hdf5read(fileName, '/FT'); feat = feat(:,:,:,(i-1)*28+1:(i-1)*28+28);
% im = hdf5read(fileName, '/IN'); im = im(:,:,:,(i-1)*28+1:(i-1)*28+28);
% dx = hdf5read(fileName, '/DX'); dx = dx((i-1)*28+1:(i-1)*28+28);
% dy = hdf5read(fileName, '/DY'); dy = dy((i-1)*28+1:(i-1)*28+28);
% gt = hdf5read(fileName, '/GT'); gt = gt(:,:,:,(i-1)*28+1:(i-1)*28+28);
% input_data{1} = permute(feat, [2 1 3 4]);
% input_data{2} = reshape(dx, [1 1 1 length(dx)]);
% input_data{3} = reshape(dy, [1 1 1 length(dy)]); 
% 
% % Initialize a network
% net = caffe.Net(net_model, net_weights, phase);
% 
% % tic;
% output = net.forward(input_data);
% output = output{1};
% output = permute(output, [2 1 3 4]);
% % toc;
% % figure;imagesc(output);
% fe(ff) = fe(ff) + sum(sum(sum(sum((gt-output).^2))))/2;
% caffe.reset_all();
% end
% fe(ff)
% ff = ff+1;
% end

