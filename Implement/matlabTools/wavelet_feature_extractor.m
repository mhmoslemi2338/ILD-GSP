


function vector=wavelet_feature_extractor(I)
    
    % wavelet Parameters
    filterlen = 24; % filter length
    nnz_factor = 1; % fraction of non-zero coefficient
    max_level = 2; % number of decomposition levels
    theta = 2; % number of bipartite graphs
    mystep=4; 


    Data=double(I);
    [Gs ,N ,Ln_bpt ,Colorednodes ,beta_dist] = define_graph(Data , theta , max_level);
    f_w = wavelet_response(Data,N, Ln_bpt, Colorednodes, beta_dist, filterlen, theta);

    % convert wavelet response to 4D matrix and save
    vector=[];
    for i=1:max_level
        wavelet_level=f_w{i};
        dim=sqrt(length(wavelet_level));
        wavelet_level_sq=zeros(dim,dim,4);
        for j=1:4
            wavelet_band=wavelet_level(:,j);
            wavelet_band=reshape(wavelet_band,[dim,dim]);
            wavelet_level_sq(:,:,j)=wavelet_band;
        end     


        for idx=1:4
            mymat=wavelet_level_sq(:,:,idx);
            [m,n]=size(mymat);
            F1=[];
            F2=[];
            F3=[];  
            for ii=1:m-mystep+1
                for jj=1:n-mystep+1
                    window=mymat(ii:ii+mystep-1,jj:jj+mystep-1);
                    singularValue=svd(window);
                    F1=[F1; max(singularValue)];
                    F2=[F2; mean(singularValue)];
                    F3=[F3; median(singularValue)];
                end
            end
            vector=[vector wblfit(F1) wblfit(F2) wblfit(F3)];
        end
    end

end
