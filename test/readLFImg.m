function video = readLFImg(path)

    global parameters;
    h = parameters.h;
    w = parameters.w;
    uv_dia = parameters.uv_dia;
    
    tic;
    fprintf('reading LF image... '); 
    
    frameList = dir([path '/*eslf.png']);                           % list of LF frames
    fNum = size(frameList, 1);                                      % total number of frames
    video = zeros(376, 541, 3, uv_dia, uv_dia, fNum, 'single');     % input ground truth video
    for f = 1:fNum
        LF_frame = im2single(imread([path '/' frameList(f).name]));
        video(:,:,:,:,:,f) = permute(reshape(LF_frame, [uv_dia 376 uv_dia 541 3]), [2 4 5 1 3]);
    end    
    video = imresize(video, [h w]);
    video = max(0, log(max(1e-5,video))/5 + 1);
    
    fprintf('done in %.2f secs\n', toc); 
end