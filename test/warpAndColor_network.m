function video_output = warpAndColor_network(net, video, ...
    flow_02t, flow_T2t, disp_0, disp_T)           

    global parameters;
    h = parameters.h;
	w = parameters.w;
    uv_dia = parameters.uv_dia;
    vc = parameters.vc;
    uc = parameters.uc; 
    T = parameters.T;
    meanIm = parameters.meanIm;
    
    video_output = zeros(h, w, 3, uv_dia, uv_dia, T-1, 'uint8');     % output LF video of our method
    
    %% generate input to the network
    input_data{1} = preprocess(video(:,:,:,vc,uc,1));                % im_0
    input_data{3} = preprocess(video(:,:,:,vc,uc,T+1));              % im_T
    input_data{8} = permute(disp_0, [2 1 3]);                        % disp_0
    input_data{9} = permute(disp_T, [2 1 3]);                        % disp_T
    for t = 1:T-1                                              % for all intermediate frames        
        msg = sprintf('frame %d of %d frames', t, T-1);
        fprintf(msg);
        input_data{5} = preprocess(video(:,:,:,vc,uc,t+1));          % im_t
        input_data{6} = permute(flow_02t(:,:,:,t), [2 1 3]);         % flow_02t
        input_data{7} = permute(flow_T2t(:,:,:,t), [2 1 3]);         % flow_T2t
        input_data{10} = t/T;                                        % lambda  
        for v = 1:uv_dia                                       % for all angular views
            for u = 1:uv_dia
                input_data{2} = preprocess(video(:,:,:,v,u,1));      % lf_0
                input_data{4} = preprocess(video(:,:,:,v,u,T+1));    % lf_T   
                input_data{11} = (u-uc) / (uv_dia/2);          % normalized u
                input_data{12} = (v-vc) / (uv_dia/2);          % normalized v
                
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