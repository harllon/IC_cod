function video_output = warpAndColor_network(net, video_2D, flow_lf0_to_imt, flow_lfT_to_imt, ...
    flow_im0_to_imt, flow_imT_to_imt, disp_0, disp_T)
                  
    global parameters;
    h = parameters.h;
	w = parameters.w;
    uv_dia = parameters.uv_dia;
    vc = parameters.vc;
    uc = parameters.uc; 
    T = parameters.T;
    offset = parameters.offset;
    meanIm = parameters.meanIm;
    
    video_output = zeros(h, w, 3, uv_dia, uv_dia, T, 'uint8');       % output LF video of our method
    
    %% generate input to the network
    input_data{1} = preprocess(video_2D(:,:,:,offset+1));            % im_0
    input_data{2} = preprocess(video_2D(:,:,:,offset+T+1));          % im_T
    input_data{8} = permute(disp_0, [2 1 3]);                        % disparity at frame 0
    input_data{9} = permute(disp_T, [2 1 3]);                        % disparity at frame T
    for t = 1:T                                                % for all intermediate frames        
        msg = sprintf('frame %d of %d frames', t, T);
        fprintf(msg);
        input_data{3} = preprocess(video_2D(:,:,:,offset+t));        % im_t
        input_data{4} = permute(flow_lf0_to_imt(:,:,:,t), [2 1 3]);  % flow that warps lf_0 to im_t
        input_data{5} = permute(flow_lfT_to_imt(:,:,:,t), [2 1 3]);  % flow that warps lf_T to im_t
        input_data{6} = permute(flow_im0_to_imt(:,:,:,t), [2 1 3]);  % flow that warps im_0 to im_t
        input_data{7} = permute(flow_imT_to_imt(:,:,:,t), [2 1 3]);  % flow that warps im_T to im_t
        input_data{10} = t/T;                                        % lambda
        for v = 1:uv_dia                                       % for all angular views
            for u = 1:uv_dia
                input_data{11} = (u-uc) / (uv_dia/2);                % normalized u
                input_data{12} = (v-vc) / (uv_dia/2);                % normalized v
                
                % warp flow + color estimation
                output_data = net.forward(input_data);    
                video_output(:,:,:,v,u,t) = uint8((permute(output_data{1}, [2 1 3]) + meanIm)*255);
            end
        end    
        fprintf(repmat(char(8), [1 numel(msg)]));
    end
end

function in_processed = preprocess(input)
    global parameters;        
    meanIm = parameters.meanIm;
    in_processed = permute(input-meanIm, [2 1 3]);
end
    