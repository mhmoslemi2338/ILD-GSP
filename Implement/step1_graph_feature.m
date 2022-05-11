

clear all; clc; close all


addpath matlabTools/gspbox/
addpath matlabTools/
gsp_start

cnt=1;
dest_dir='texture_features';
lables=split(strip(ls('data/')));

% prepare folders
if ~ isdir(dest_dir)
    mkdir(dest_dir);
end

path={};
for k=1:length(lables)
    ll=lables{k};
    path=[path; join(['data/',ll])];
    if ~ isdir(join([dest_dir,'/',ll]))
        mkdir(join([dest_dir,'/',ll]))
    end
end


for k=1:length(path)
    ll=path{k};
    images=ls(ll);
    images=split(strip(images));
    for k2=1:length(images)
        filename=join([ll,'/',images{k2}]);
        img = uint8(imread(filename));
 
        %%%%%% HVG , IVG
        feature_vector=[];
        for method={'horizontal','natural'}
            for lattice={true , false}
                for I = {img ,255-img} 
                    I=cell2mat(I);
                    Edge_list=imageVisibilityGraph(I,cell2mat(method),cell2mat(lattice));
                    G = graph(Edge_list(:,1),Edge_list(:,2));
                    Deg_seq = degree(G);
                    N=size(I,1).^2;
                    % Degree distribution P(k)
                    Pk=hist(Deg_seq,1:1:max(Deg_seq))./N; Pk(Pk==0)=10^-20;
                    Z=visibilityPatches(I,1,cell2mat(method));
                    feature_vector=[feature_vector wblfit(Pk) Z];
                end
            end
        end   
        %%%%%% wavelet
        wavelet_feature=wavelet_feature_extractor(img);
        feature_vector=[feature_vector wavelet_feature];

        savepath=replace(replace(filename,'.png','.mat'),'data',dest_dir);
        save(savepath,'feature_vector');
        %print progress
        cnt=cnt+1;
        if mod(cnt,296)==0
            disp(join(['progress:',num2str(cnt/296),' % ']))
        end                 
    end     
end




