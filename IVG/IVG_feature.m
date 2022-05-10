

clear all
clc
close all


cnt=1;


for dest_dir={'IVG_I_lattice','IVG_2I_lattice','HVG_I_lattice','HVG_2I_lattice'}
    dest_dir=cell2mat(dest_dir);


    % prepare folders
    if ~ isdir(dest_dir)
        mkdir(dest_dir);
    end
    if ~ isdir(join([dest_dir,'/Train_ILD']))
        mkdir(join([dest_dir,'/Train_ILD']));
    end
    if ~ isdir(join([dest_dir,'/Test_Talisman']))
        mkdir(join([dest_dir,'/Test_Talisman']));
    end

end



data_type={'Train_ILD','Test_Talisman'};
data_type={'Train_ILD'};

for idx=1:2
    typee=data_type{idx};
    lables=ls(join(['/Users/mohammad/Desktop/Bsc prj/Graph wavelet implement/data/',typee]));
    lables=(split(strip(lables)));
    
    

    for dest_dir={'IVG_I_lattice','IVG_2I_lattice','HVG_I_lattice','HVG_2I_lattice'}
        dest_dir=cell2mat(dest_dir);
        path={};
        for k=1:length(lables)
            ll=lables{k};
            path=[path; join(['/Users/mohammad/Desktop/Bsc prj/Graph wavelet implement/data/',typee,'/',ll])];
            if ~ isdir(join([dest_dir,'/',typee,'/',ll]))
                mkdir(join([dest_dir,'/',typee,'/',ll]))
            end
        end
    end




    for k=1:length(path)
        ll=path{k};
        images=ls(ll);
        images=split(strip(images));
    
         for k2=1:length(images)

            filename=join([ll,'/',images{k2}]);

    %%%%%%%% HVG
            % original image
            I = uint8(imread(filename));
            Edge_list=imageVisibilityGraph(I,'horizontal',true);
            G = graph(Edge_list(:,1),Edge_list(:,2));
            Deg_seq = degree(G);
            N=size(I,1).^2;
            % Degree distribution P(k)
            Pk=hist(Deg_seq,1:1:max(Deg_seq))./N; Pk(Pk==0)=10^-20;
            Z=visibilityPatches(I,1,'horizontal');
            feature_vector=[wblfit(Pk) Z];
            savepath=replace(filename,'.png','.mat');
            savepath=replace(savepath,'/Users/mohammad/Desktop/Bsc prj/Graph wavelet implement/data','HVG_I_lattice');
            save(savepath,'feature_vector'); %save wavelet response
            % reverse image
            I=255-I;
            Edge_list=imageVisibilityGraph(I,'horizontal',true);
            G = graph(Edge_list(:,1),Edge_list(:,2));
            Deg_seq = degree(G);
            N=size(I,1).^2;
            % Degree distribution P(k)
            Pk=hist(Deg_seq,1:1:max(Deg_seq))./N; Pk(Pk==0)=10^-20;
            Z=visibilityPatches(I,1,'horizontal');
            feature_vector=[feature_vector wblfit(Pk) Z];
            savepath=replace(filename,'.png','.mat');
            savepath=replace(savepath,'/Users/mohammad/Desktop/Bsc prj/Graph wavelet implement/data','HVG_2I_lattice');
            save(savepath,'feature_vector'); %save wavelet response
                    
     

     %%%%%%%% IVG
            % original image
            I = uint8(imread(filename));
            Edge_list=imageVisibilityGraph(I,'natural',true);
            G = graph(Edge_list(:,1),Edge_list(:,2));
            Deg_seq = degree(G);
            N=size(I,1).^2;
            % Degree distribution P(k)
            Pk=hist(Deg_seq,1:1:max(Deg_seq))./N; Pk(Pk==0)=10^-20;
            Z=visibilityPatches(I,1,'natural');
            feature_vector=[wblfit(Pk) Z];
            savepath=replace(filename,'.png','.mat');
            savepath=replace(savepath,'/Users/mohammad/Desktop/Bsc prj/Graph wavelet implement/data','IVG_I_lattice');
            save(savepath,'feature_vector'); %save wavelet response           

            % reverse image
            I=255-I;
            Edge_list=imageVisibilityGraph(I,'natural',true);
            G = graph(Edge_list(:,1),Edge_list(:,2));
            Deg_seq = degree(G);
            N=size(I,1).^2;
            % Degree distribution P(k)
            Pk=hist(Deg_seq,1:1:max(Deg_seq))./N; Pk(Pk==0)=10^-20;
            Z=visibilityPatches(I,1,'natural');
            feature_vector=[feature_vector wblfit(Pk) Z];
            savepath=replace(filename,'.png','.mat');
            savepath=replace(savepath,'/Users/mohammad/Desktop/Bsc prj/Graph wavelet implement/data','IVG_2I_lattice');
            save(savepath,'feature_vector'); %save wavelet response
             


            %print progress
            cnt=cnt+1;
            if mod(cnt,485)==0
                disp(join(['progress:',num2str(cnt/485),' % ']))
            end                 
        end     
    end
end



