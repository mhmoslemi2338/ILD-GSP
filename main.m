

clear all
clc
close all
cnt=0;



addpath tools/sgwt_toolbox/
addpath tools/Graph_kernels/
addpath tools/gspbox/
addpath tools/toolbox/
gsp_start



% Parameters
filterlen = 24; % filter length
nnz_factor = 1; % fraction of non-zero coefficient
max_level = 2; % number of decomposition levels
theta = 2; % number of bipartite graphs



% prepare folders
if ~ isdir('wavelet_mat')
    mkdir('wavelet_mat');
end
if ~ isdir('wavelet_mat/Train_ILD')
    mkdir('wavelet_mat/Train_ILD');
end
if ~ isdir('wavelet_mat/Test_Talisman')
    mkdir('wavelet_mat/Test_Talisman');
end




data_type={'Train_ILD','Test_Talisman'};
for idx=1:2
    typee=data_type{idx};

    
    lables=ls(join(['data/',typee]));
    lables=(split(strip(lables)));
    
    
    path={};
    for k=1:length(lables)
        ll=lables{k};
        path=[path; join(['data/',typee,'/',ll])];
        if ~ isdir(join(['wavelet_mat/',typee,'/',ll]))
            mkdir(join(['wavelet_mat/',typee,'/',ll]))
        end
    end

    for k=1:length(path)
        ll=path{k};
        images=ls(ll);
        images=split(strip(images));
    
        for k2=1:length(images)
            filename=join([ll,'/',images{k2}]);

    
            % Graph Signal
            Data = imread(filename);
            Data = double(Data);
            [Gs ,N ,Ln_bpt ,Colorednodes ,beta_dist] = define_graph(Data , theta , max_level);
            f_w = wavelet_response(Data,N, Ln_bpt, Colorednodes, beta_dist, filterlen, theta);

            % convert wavelet response to 4D matrix and save
            for i=1:max_level
                wavelet_level=f_w{i};
                dim=sqrt(length(wavelet_level));
                wavelet_level_sq=zeros(dim,dim,4);
                for j=1:4
                    wavelet_band=wavelet_level(:,j);
                    wavelet_band=reshape(wavelet_band,[dim,dim]);
                    wavelet_level_sq(:,:,j)=wavelet_band;
                end     

                savepath=replace(filename,'.png',join(['_level',num2str(i),'.mat']));
                savepath=replace(savepath,'data','wavelet_mat');
                save(savepath,'wavelet_level_sq'); %save wavelet response
                
                %print progress
                cnt=cnt+1;
                if mod(cnt,485)==0
                    disp(join(['progress:',num2str(cnt/485),' % ']))
                end
            end             
        end     
    end


end


%% plot wavelet response
% close all
% plot_wavelet_response(f_w{1},32,1)
% plot_wavelet_response(f_w{2},16,2)
% 

