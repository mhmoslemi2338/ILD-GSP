

clear all
clc
close all


cnt=1;


dest_dir='IVG_feature';




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




data_type={'Train_ILD','Test_Talisman'};


for idx=1:2
    typee=data_type{idx};
    lables=ls(join(['/Users/mohammad/Desktop/Bsc prj/Graph wavelet implement/data/',typee]));
    lables=(split(strip(lables)));
    
    
    path={};
    for k=1:length(lables)
        ll=lables{k};
        path=[path; join(['/Users/mohammad/Desktop/Bsc prj/Graph wavelet implement/data/',typee,'/',ll])];
        if ~ isdir(join([dest_dir,'/',typee,'/',ll]))
            mkdir(join([dest_dir,'/',typee,'/',ll]))
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
            Edge_list=imageVisibilityGraph(I,'horizontal',false);
            G = graph(Edge_list(:,1),Edge_list(:,2));
            Deg_seq = degree(G);
            N=size(I,1).^2;
            % Degree distribution P(k)
            Pk=hist(Deg_seq,1:1:max(Deg_seq))./N; Pk(Pk==0)=10^-20;
            Z=visibilityPatches(I,1,'horizontal');
            feature_vector=[wblfit(Pk) Z];
            % reverse image
            I=255-I;
            Edge_list=imageVisibilityGraph(I,'horizontal',false);
            G = graph(Edge_list(:,1),Edge_list(:,2));
            Deg_seq = degree(G);
            N=size(I,1).^2;
            % Degree distribution P(k)
            Pk=hist(Deg_seq,1:1:max(Deg_seq))./N; Pk(Pk==0)=10^-20;
            Z=visibilityPatches(I,1,'horizontal');
            feature_vector=[feature_vector wblfit(Pk) Z];
        
% % %     %%%%%%%% IVG
% % %             % original image
% % %             I = uint8(imread(filename));
% % %             Edge_list=imageVisibilityGraph(I,'horizontal',true);
% % %             G = graph(Edge_list(:,1),Edge_list(:,2));
% % %             Deg_seq = degree(G);
% % %             N=size(I,1).^2;
% % %             % Degree distribution P(k)
% % %             Pk=hist(Deg_seq,1:1:max(Deg_seq))./N; Pk(Pk==0)=10^-20;
% % %             Z=visibilityPatches(I,1,'horizontal');
% % %             Z=Z(:,1:2:end)+Z(:,2:2:end);
% % %             feature_vector=[feature_vector wblfit(Pk) Z];
% % %             % reverse image
% % %             I=255-I;
% % %             Edge_list=imageVisibilityGraph(I,'horizontal',true);
% % %             G = graph(Edge_list(:,1),Edge_list(:,2));
% % %             Deg_seq = degree(G);
% % %             N=size(I,1).^2;
% % %             % Degree distribution P(k)
% % %             Pk=hist(Deg_seq,1:1:max(Deg_seq))./N; Pk(Pk==0)=10^-20;
% % %             Z=visibilityPatches(I,1,'horizontal');
% % %             Z=Z(:,1:2:end)+Z(:,2:2:end);
% % %             feature_vector=[feature_vector wblfit(Pk) Z];
% % %             
            savepath=replace(filename,'.png','.mat');
            savepath=replace(savepath,'/Users/mohammad/Desktop/Bsc prj/Graph wavelet implement/data',dest_dir);
            save(savepath,'feature_vector'); %save wavelet response
             
            %print progress
            cnt=cnt+1;
            if mod(cnt,485)==0
                disp(join(['progress:',num2str(cnt/485),' % ']))
            end                 
        end     
    end
end



