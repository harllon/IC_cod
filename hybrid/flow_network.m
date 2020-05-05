function [flow_lf0_to_imt, flow_lfT_to_imt, flow_im0_to_imt, flow_imT_to_imt] = ...
    flow_network(net, net_hybrid, video_lf, video_2D, fn)

    global parameters;
    T = parameters.T;
    h = parameters.h;
    w = parameters.w;
    vc = parameters.vc;
    uc = parameters.uc;
    meanIm = parameters.meanIm;
    offset = parameters.offset;
    use_sequential = 1;
    
    flow_lf0_to_imt = zeros(h, w, 2, T+1, 'single'); % flows that warp lf_0 (central view of first LF) to im_t (2D frame t)
    flow_lfT_to_imt = zeros(h, w, 2, T+1, 'single');
    flow_im0_to_imt = zeros(h, w, 2, T+1, 'single'); % flows that warp im_0 (first 2D frame) to im_t (2D frame t)
    flow_imT_to_imt = zeros(h, w, 2, T+1, 'single');

    im_0 = video_2D(:,:,:,offset+1);
    im_T = video_2D(:,:,:,offset+T+1);
    lf_0 = video_lf(:,:,:,vc,uc,fn);
    lf_T = video_lf(:,:,:,vc,uc,fn+1);        
    flow_lf0_to_imt(:,:,:,1) = net_forward(im_0, lf_0, net_hybrid, meanIm);   % flow that warps lf_0 to im_0
    flow_lfT_to_imt(:,:,:,end) = net_forward(im_T, lf_T, net_hybrid, meanIm); % flow that warps lf_T to im_T
                             
    if (~use_sequential)     % method 1: directly estimate flows between frame 0/T and frame t
        for t = 1:T+1
            im_t = video_2D(:,:,:,offset+t);
            flow_0 = net_forward(im_t, im_0, net, meanIm);
            flow_T = net_forward(im_t, im_T, net, meanIm);
            flow_im0_to_imt(:,:,:,t) = flow_0 * (t ~= 1);
            flow_imT_to_imt(:,:,:,t) = flow_T * (t ~= (T+1));
            flow_lf0_to_imt(:,:,:,t) = flow_0 + warpImage(flow_lf0_to_imt(:,:,:,1), flow_0);
            flow_lfT_to_imt(:,:,:,t) = flow_T + warpImage(flow_lfT_to_imt(:,:,:,end), flow_T);
        end
    else                     % method 2: sequentially estimate flow between neighboring frames and propagate
        for t = 1:T
            im_t = video_2D(:,:,:,offset+t+1);
            im_0 = video_2D(:,:,:,offset+t);
            flow_0 = net_forward(im_t, im_0, net, meanIm);

            im_t = video_2D(:,:,:,offset+T-t+1);
            im_T = video_2D(:,:,:,offset+T-t+2);
            flow_T = net_forward(im_t, im_T, net, meanIm);                   

            flow_im0_to_imt(:,:,:,t+1) = flow_0 + warpImage(flow_im0_to_imt(:,:,:,t), flow_0);
            flow_imT_to_imt(:,:,:,T-t+1) = flow_T + warpImage(flow_imT_to_imt(:,:,:,T-t+2), flow_T);
            flow_lf0_to_imt(:,:,:,t+1) = flow_0 + warpImage(flow_lf0_to_imt(:,:,:,t), flow_0);
            flow_lfT_to_imt(:,:,:,T-t+1) = flow_T + warpImage(flow_lfT_to_imt(:,:,:,T-t+2), flow_T);
        end  
    end
            
end

function flow = net_forward(im1, im2, net, meanIm)
    input_data{1} = permute(im1-meanIm, [2 1 3]);
    input_data{2} = permute(im2-meanIm, [2 1 3]);
    output_data = net.forward(input_data);
    flow = permute(output_data{1}, [2 1 3]);
end