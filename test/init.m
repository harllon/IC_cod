h = 352;      % output video height
w = 512;      % output video width
uv_dia = 8;   % angular resolution
uc = 4;       % angular index for central view
vc = 4;       % angular index for central view
meanIm = repmat(reshape([0.4079, 0.4575, 0.4811], [1 1 3]), [h w 1]); % mean img for data preprocessing

global parameters
parameters.h = h;
parameters.w = w;
parameters.uv_dia = uv_dia;
parameters.uc = uc;
parameters.vc = vc;
parameters.meanIm = meanIm;