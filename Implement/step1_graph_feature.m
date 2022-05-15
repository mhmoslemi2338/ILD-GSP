

clear all; clc; close all
dest_dir='texture_features';

%*************************************
%%%%%%%%%%% Prepare folders %%%%%%%%%%
%*************************************
addpath matlabTools/
cnt=1;
lables=split(strip(ls('data/')));
% prepare folders
if ~ isdir(dest_dir)
    mkdir(dest_dir);
end
if ~ isdir(join([dest_dir,'/graph']))
    mkdir(join([dest_dir,'/graph']));
end

path={};
for k=1:length(lables)
    ll=lables{k};
    path=[path; join(['data/',ll])];
    if ~ isdir(join([dest_dir,'/graph/',ll]))
        mkdir(join([dest_dir,'/graph/',ll]))
    end
end
%*************************************
%%%%%%%%%%% EXTRACT FEATURES %%%%%%%%%
%*************************************

for k=1:length(path)
    ll=path{k};
    images=split(strip(ls(ll)));
    for k2=1:length(images)
        filename=join([ll,'/',images{k2}]);
        img = uint8(imread(filename));
 
        %%%%%% HVG , IVG
        feature_vector=[];
        symmetric=0;
        for I = {img ,255-img}
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
                    global_feature=global_feature_extract(G);
                    Z=visibilityPatches(I,1,cell2mat(method));
                    feature_vector=[feature_vector global_feature Z];
                    
                
                end
            end
            %%%%%% wavelet
%             wavelet_feature=wavelet_feature_extractor(I);
%             feature_vector=[feature_vector wavelet_feature];
        end  
       
        
        
        savepath=replace(replace(filename,'.png','.mat'),'data',join([dest_dir,'/graph']));
        save(savepath,'feature_vector');
        %print progress
        cnt=cnt+1;
        if mod(cnt,148)==0
            disp(join(['progress:',num2str(cnt/296),' % ']))
        end                 
    end     
end







