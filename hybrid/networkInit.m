function [disp_net, flow_net, flow_hybrid_net, warpAndColor_net] = networkInit()

addpath('../model/caffe/matlab/');
caffe.reset_all();
caffe.set_mode_gpu();
gpu_id = 0;
caffe.set_device(gpu_id);

model_dir = '../model/';
net_weights = [model_dir 'LF_video.caffemodel'];
disp_model = [model_dir 'deploy_disp.prototxt'];
flow_model = [model_dir 'deploy_flow.prototxt'];
flow_hybrid_model = [model_dir 'deploy_flow_hybrid.prototxt'];
warpAndColor_model = [model_dir 'deploy_warpAndColor_hybrid.prototxt'];

tic;
fprintf('Loading network... ');
phase = 'test';
disp_net = caffe.Net(disp_model, net_weights, phase);
flow_net = caffe.Net(flow_model, net_weights, phase);
flow_hybrid_net = caffe.Net(flow_hybrid_model, net_weights, phase);
warpAndColor_net = caffe.Net(warpAndColor_model, net_weights, phase);
fprintf('done in %.2f secs\n', toc);
