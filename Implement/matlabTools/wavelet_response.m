

function f_w = wavelet_response(Data,N, Ln_bpt, Colorednodes, beta_dist, filterlen, theta)

    [max_level ,~ ]=size(Ln_bpt);
    Fmax = 2^theta; % Graph Coloring

    % design a low-pass kernel
    g = @(x)(meyer_kernel(x));  % Meyer Wavelet KernelF
    arange = [0 2];
    c=sgwt_cheby_coeff(g,filterlen,filterlen+1,arange);
    % Compute Filterbank Output at each channel
    f = Data(:);
    f = f/norm(f);
    % Compute Filterbank Output at each channel
    f_w = cell(max_level,1);
    Channel_Name = cell(max_level,Fmax);
    for level = 1:max_level
        f_w{level} = zeros(N(level)/(2^(level-1)),Fmax);
        for i = 1:Fmax
            if level == 1
                tempf_w = f;
            else
                tempf_w = f_w{level-1}(Colorednodes{level-1,1},1);
            end
            for j = 1: theta
                if beta_dist{level}(i,j) == 1
                    tempf_w = sgwt_cheby_op(tempf_w,Ln_bpt{level,j},c,arange);
                    Channel_Name{level,i} = strcat(Channel_Name{level,i},'L');
                else
                    tempf_w = sgwt_cheby_op(tempf_w,2*speye(N(level)) - Ln_bpt{level,j},c,arange);
                    Channel_Name{level,i} = strcat(Channel_Name{level,i},'H');
                end
            end
            f_w{level}(Colorednodes{level,i},i) = tempf_w(Colorednodes{level,i});
        end
    end

end
