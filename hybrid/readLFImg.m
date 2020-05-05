function [LFvideo, video, LFfNum, fNum, key_frame] = readLFImg(path)

    global parameters;
    h = parameters.h;
    w = parameters.w;
    uv_dia = parameters.uv_dia;
    
    %% read LF frame
    tic;
    fprintf('Loading LF images... ');
    LFframeList = dir([path '/*eslf.png']);                           % list of LF frames
    LFfNum = size(LFframeList, 1);                                    % number of LF frames
    LFvideo = zeros(h, w, 3, uv_dia, uv_dia, LFfNum, 'single');
    for t = 1:LFfNum        
        LFvideo(:,:,:,:,:,t) = readLFframe(path, LFframeList, t);
    end        
    fprintf('done in %.2f secs\n', toc);

    %% read video frame  
    tic;
    fprintf('Loading 2D video... ');
    videoObj = VideoReader([path '/video.MOV']);
    load([path '/key_frame.mat']);                          % specify frame idx of key frames
    fNum = key_frame(end) - key_frame(1) + 1;               % total number of frames we're interested in
    video = zeros(h, w, 3, fNum, 'single');
    for t = 1:fNum        
        video(:,:,:,t) = im2single(read(videoObj, key_frame(1)+t-1));
    end
    fprintf('done in %.2f secs\n', toc);
end


function LF_frame = readLFframe(path, LFframeList, t)
    global parameters;
    uv_dia = parameters.uv_dia;
    h = parameters.h;
    w = parameters.w;
    
    LF_frame = im2single(imread([path '/' LFframeList(t).name]));
    h1 = size(LF_frame, 1) / uv_dia; 
    w1 = size(LF_frame, 2) / uv_dia;
    LF_frame = log(max(1e-5, LF_frame))/5 + 1;
    LF_frame = permute(reshape(LF_frame, [uv_dia h1 uv_dia w1 3]), [2 4 5 1 3]);
    LF_frame = max(0, min(1, imresize(LF_frame, [h w])));
end