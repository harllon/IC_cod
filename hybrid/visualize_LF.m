function [v_temp, v_ang] = visualize_LF(video_output)

global parameters;
h = parameters.h;
w = parameters.w;
uv_dia = parameters.uv_dia;

% temporal video for four corner views
v_temp = squeeze([video_output(:,:,:,1,1,:) video_output(:,:,:,1,uv_dia,:);
            video_output(:,:,:,uv_dia,1,:) video_output(:,:,:,uv_dia,uv_dia,:)]);

% angular video for first 8 frames
N = 8;
v_ang = zeros(h, w, 3, N*uv_dia^2, 'uint8');
i = 1;
for t = 1:N
    for v = 1:uv_dia
        if (mod(v,2)) ustart = 1; ustep = 1; uend = uv_dia;
        else ustart = uv_dia; ustep = -1; uend = 1; end
        for u = ustart:ustep:uend                
           v_ang(:,:,:,i) = video_output(:,:,:,v,u,t);
           i = i+1;
        end
    end
    for u = 1:uv_dia
        if (mod(u,2)) vstart = uv_dia; vstep = -1; vend = 1;
        else vstart = 1; vstep = 1; vend = uv_dia; end
        for v = vstart:vstep:vend                
            v_ang(:,:,:,i) = video_output(:,:,:,v,u,t);
            i = i+1;
        end
    end
end
