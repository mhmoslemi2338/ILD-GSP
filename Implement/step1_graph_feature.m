

clear all; clc; close all
dest_dir='texture_features2';

%--------------------------------------
%---------- Prepare folders -----------
%--------------------------------------
addpath matlabTools/
cnt=1;
lables=split(strip(ls('data/')));
% prepare folders
if ~ isdir(dest_dir)
    mkdir(dest_dir);
end
if ~ isdir(join([dest_dir,'/']))
    mkdir(join([dest_dir,'/']));
end

path={};
for k=1:length(lables)
    ll=lables{k};
    path=[path; join(['data/',ll])];
    if ~ isdir(join([dest_dir,'/',ll]))
        mkdir(join([dest_dir,'/',ll]))
    end
end

%--------------------------------------
%---------- EXTRACT FEATURES ----------
%--------------------------------------
for k=1:length(path)
    ll=path{k};
    images=split(strip(ls(ll)));
    for k2=1:length(images)
        filename=join([ll,'/',images{k2}]);
        img = imread(filename);
        %%%%%% HVG , IVG %%%%%%
        feature_vector=[];
        symmetric=0;
        for I = {uint8(img) ,255-uint8(img)}
            symmetric=symmetric+1;
            I=cell2mat(I);
            for method={'horizontal','natural'}
                for lattice={true , false}    
                    if (symmetric==2) && (cell2mat(lattice)==false)
                        continue
                    end
                    if (symmetric==1) && (length(cell2mat(method))==7) && (cell2mat(lattice)==true)
                        continue
                    end
                    Edge_list=imageVisibilityGraph(I,cell2mat(method),cell2mat(lattice));
                    G = graph(Edge_list(:,1),Edge_list(:,2));
                    Z=visibilityPatches(I,1,cell2mat(method));
                    feature_vector=[feature_vector Z];
                end
            end
        end  
        %%%%%% wavelet %%%%%%
        wavelet_feature1=wavelet_feature_extractor(double(img));
        wavelet_feature2=wavelet_feature_extractor(255-double(img));
        feature_vector=[feature_vector wavelet_feature1 wavelet_feature2];
        %%%%% Saving result %%%%%%
        savepath=replace(replace(filename,'.png','.mat'),'data',dest_dir);
        save(savepath,'feature_vector');
        %%%%% print progress %%%%%%
        cnt=cnt+1;
        if mod(cnt,296)==0
            disp(join(['progress:',num2str(cnt/296),' % ']))
        end                 
    end     
end

