function disparity = disp_network(net, video)

    global parameters;
    h = parameters.h;
    w = parameters.w;
    uv_dia = parameters.uv_dia;
    meanIm = parameters.meanIm;

    input_data{1} = zeros(w, h, 3*uv_dia^2, 'single');
    input_data{2} = zeros(1, 1, uv_dia^2, 'single');
    input_data{3} = zeros(1, 1, uv_dia^2, 'single');  
    for v = 1:uv_dia
        for u = 1:uv_dia
            uv_idx = (v-1)*uv_dia + u-1 ;
            input_data{1}(:,:,uv_idx*3+1:uv_idx*3+3) = permute(video(:,:,:,v,u)-meanIm, [2 1 3]);
            input_data{2}(uv_idx+1) = u - uv_dia/2;
            input_data{3}(uv_idx+1) = v - uv_dia/2;
        end
    end
    output_data = net.forward(input_data);
    disparity = permute(output_data{1}, [2 1 3]);