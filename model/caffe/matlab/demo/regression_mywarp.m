% function output = regression_my(input_data)
% clear all

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
net_model = [model_dir 'deploy_warp.prototxt'];
net_weights = [model_dir 'model/warp_iter_10000.caffemodel'];
phase = 'test'; % run with phase test (so that dropout isn't applied)
if ~exist(net_weights, 'file')
  error('Please download CaffeNet from Model Zoo before you run this demo');
end

% fileName = '/media/C/Users/tcwang/Desktop/Code/2_TrainingData/ProcessedTrainingStage1-Lytro/1/training.h5';
fileName = '/media/C/Users/tcwang/Desktop/Code/2_TrainingData/ProcessedTrainingStage1-Lytro/1/IMG_1468/training.h5';
feat = hdf5read(fileName, '/FT'); feat = feat(:,:,:,1:60);
% im = hdf5read(fileName, '/IN'); im = im(:,:,:,1:60);
dx = hdf5read(fileName, '/DX'); dx = dx(1:60);
dy = hdf5read(fileName, '/DY'); dy = dy(1:60);
dx2 = hdf5read(fileName, '/DX2'); dx2 = dx2(1:60);
dy2 = hdf5read(fileName, '/DY2'); dy2 = dy2(1:60);
dx3 = hdf5read(fileName, '/DX3'); dx3 = dx3(1:60);
dy3 = hdf5read(fileName, '/DY3'); dy3 = dy3(1:60);
dx4 = hdf5read(fileName, '/DX4'); dx4 = dx4(1:60);
dy4 = hdf5read(fileName, '/DY4'); dy4 = dy4(1:60);
gt = hdf5read(fileName, '/GT'); gt = gt(:,:,:,1:60);
input_data{1} = permute(feat, [1 2 3 4]);
% input_data{2} = permute(im, [1 2 3 4]);
% input_data{2} = permute(gt, [2 1 3 4]);
input_data{2} = reshape(dx, [1 1 1 length(dx)]);
input_data{3} = reshape(dy, [1 1 1 length(dy)]); 
input_data{4} = reshape(dx2, [1 1 1 length(dx)]);
input_data{5} = reshape(dy2, [1 1 1 length(dy)]); 
input_data{6} = reshape(dx3, [1 1 1 length(dx)]);
input_data{7} = reshape(dy3, [1 1 1 length(dy)]); 
input_data{8} = reshape(dx4, [1 1 1 length(dx)]);
input_data{9} = reshape(dy4, [1 1 1 length(dy)]); 

% Initialize a network
net = caffe.Net(net_model, net_weights, phase);

tic;
output = net.forward(input_data);
% output = output{1};
% output = permute(output, [1 2 3 4]);
toc;
caffe.reset_all();
% figure;imagesc(output);

