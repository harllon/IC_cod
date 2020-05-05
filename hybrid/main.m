clear all
computeAllFrames = 0;   % 0 for computing a sample key frame, 1 for computing all key frames

%% initialization
init();
[disp_net, flow_net, flow_hybrid_net, warpAndColor_net] = networkInit();
global parameters;
h = parameters.h;             % image height
w = parameters.w;             % image width
uv_dia = parameters.uv_dia;   % angular resolution

%% read in LF images
% video_lf (6D): h x w x 3 x uv_dia x uv_dia x NumberOfFrames(LFfNum)
% video_2D (4D): h x w x 3 x NumberOfFrames(fNum)
% key_frame: frame indices for the key frames
path = 'data/seq01';
[video_lf, video_2D, fNum_lf, fNum_2D, key_frame] = readLFImg(path);

%% %%%%%%%%%%%%%%%%%%%%% Network Computation %%%%%%%%%%%%%%%%%%%
if computeAllFrames
    frame_list = 1:(fNum_lf-1);                                        % for all key frames in LF
    video_output = zeros(h, w, 3, uv_dia, uv_dia, fNum_2D-1, 'uint8'); % output of our method    
else
    frame_list = 1;                                                    % the first key frame only
    fNum_out = key_frame(2) - key_frame(1);
    video_output = zeros(h, w, 3, uv_dia, uv_dia, fNum_out, 'uint8');  % output of our method 
end
for fn = frame_list
    fprintf('###### Key frame %d ######\n', fn);
    offset = key_frame(fn) - key_frame(1);             % the distance between current and first key frame
    T = key_frame(fn+1) - key_frame(fn);               % the distance between current and next key frame
    parameters.offset = offset;
    parameters.T = T;
        
    %% part1: disparity estimation (at key frames)
    tic;
    if (fn == 1)  % disparity for first frame; copy from the previous frame if computed before
        disp_0 = disp_network(disp_net, video_lf(:,:,:,:,:,fn));
    else
        disp_0 = disp_T;
    end
    disp_T = disp_network(disp_net, video_lf(:,:,:,:,:,fn+1));
    fprintf('part1: disparity estimation done in %.2f secs\n', toc);
        
    %% part2: temporal flow estimation (between neighboring 2D frames)
    tic;        
    [flow_lf0_to_imt, flow_lfT_to_imt, flow_im0_to_imt, flow_imT_to_imt] ...
        = flow_network(flow_net, flow_hybrid_net, video_lf, video_2D, fn);
    fprintf('part2: temporal flow estimation done in %.2f secs\n', toc);   
                
        
    %% part 3: warp flow estimation + color estimation    
    fprintf('part3: warp flow + color estimation ... '); 
    tic
    video_output(:,:,:,:,:,offset+1:offset+T) = ...
        warpAndColor_network(warpAndColor_net, video_2D, flow_lf0_to_imt, ...
        flow_lfT_to_imt, flow_im0_to_imt, flow_imT_to_imt, disp_0, disp_T);
    fprintf('done in %.2f secs\n', toc);        
end  

%% visualize result
[v_temp, v_ang] = visualize_LF(video_output);
implay(v_temp, 10);
implay(v_ang, 30);
