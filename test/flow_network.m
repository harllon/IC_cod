function [flow_02t, flow_T2t] = flow_network(net, video)

    global parameters;
    h = parameters.h;
    w = parameters.w;
    T = parameters.T;
    vc = parameters.vc;
    uc = parameters.uc;
    meanIm = parameters.meanIm;
    use_sequential = 1;

    flow_02t = zeros(h, w, 2, T-1); % flows which warp frame 0 to frame t's
    flow_T2t = zeros(h, w, 2, T-1); % flows which warp frame T to frame t's
    if (~use_sequential)            % method 1: directly estimate flows between frame 0/T and frame t
        im_0 = video(:,:,:,vc,uc,1); 
        im_T = video(:,:,:,vc,uc,T+1); 
        for t = 1:T-1
            im_t = video(:,:,:,vc,uc,t+1);                                
            flow_02t(:,:,:,t) = net_forward(im_t, im_0, net, meanIm);            
            flow_T2t(:,:,:,t) = net_forward(im_t, im_T, net, meanIm);
        end
        
    else                            % method 2: sequentially estimate flow between neighboring frames and propagate
        flow_02t = zeros(h, w, 2, T);
        flow_T2t = zeros(h, w, 2, T);
        for t = 1:T-1
            im_0 = video(:,:,:,vc,uc,t);  
            im_t = video(:,:,:,vc,uc,t+1);
            flow_0 = net_forward(im_t, im_0, net, meanIm);

            im_t = video(:,:,:,vc,uc,T-t+1);
            im_T = video(:,:,:,vc,uc,T-t+2);
            flow_T = net_forward(im_t, im_T, net, meanIm);

            flow_02t(:,:,:,t+1) = flow_0 + warpImage(flow_02t(:,:,:,t), flow_0);
            flow_T2t(:,:,:,T-t) = flow_T + warpImage(flow_T2t(:,:,:,T-t+1), flow_T);
        end
        flow_02t = flow_02t(:,:,:,2:end);
        flow_T2t = flow_T2t(:,:,:,1:end-1);
    end    
            
end

function flow = net_forward(im1, im2, net, meanIm)
    input_data{1} = permute(im1-meanIm, [2 1 3]);
    input_data{2} = permute(im2-meanIm, [2 1 3]);
    output_data = net.forward(input_data);
    flow = permute(output_data{1}, [2 1 3]);
end