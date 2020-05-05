clear all

%% initialization
init();                         % parameter initialization
[disp_net, flow_net, warpAndColor_net] = networkInit();  % the four networks used in our system

%% read in LF images

file_path = 'data/seq01';
video = readLFImg(file_path);   % LF ground truth video
T = size(video, 6)-1;           % temporal upsampling factor
global parameters;
parameters.T = T;

%% %%%%%%%%%%%%%%%%%%%%% Network Computation %%%%%%%%%%%%%%%%%%%

%% part1: disparity estimation (at key frames)
tic;                
disp_0 = disp_network(disp_net, video(:,:,:,:,:,1));               % disparity at frame 0
disp_T = disp_network(disp_net, video(:,:,:,:,:,T+1));             % disparity at frame T
fprintf('part1: disparity estimation done in %.2f secs\n', toc);

%% part2: temporal flow estimation (between neighboring 2D frames)
tic;
[flow_02t, flow_T2t] = flow_network(flow_net, video);              % flows which warp frame 0/T to frame t
fprintf('part2: temporal flow estimation done in %.2f secs\n', toc);        

%% part 3: warp flow estimation + color estimation
fprintf('part3: warp flow + color estimation ... '); 
tic
video_output = warpAndColor_network(warpAndColor_net, video, flow_02t, flow_T2t, disp_0, disp_T);
fprintf('done in %.2f secs\n', toc); 


%% visualize result
[v_temp, v_ang] = visualize_LF(video_output);
implay(v_temp, 10);
implay(v_ang, 30);
