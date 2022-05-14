
clear all; clc; close all
dest_dir='texture_features2';

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
if ~ isdir(join([dest_dir,'/wavelet']))
    mkdir(join([dest_dir,'/wavelet']));
end
path={};
for k=1:length(lables)
    ll=lables{k};
    path=[path; join(['data/',ll])];
    if ~ isdir(join([dest_dir,'/wavelet/',ll]))
        mkdir(join([dest_dir,'/wavelet/',ll]))
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
 
        feature_vector=[];
        for I = {img ,255-img}
            I=cell2mat(I);
            %%%%%% wavelet
            wavelet_feature=wavelet_feature_extractor(I);
            feature_vector=[feature_vector wavelet_feature];
        end   

        savepath=replace(replace(filename,'.png','.mat'),'data',join([dest_dir,'/wavelet']));
        save(savepath,'feature_vector');
        %print progress
        cnt=cnt+1;
        if mod(cnt,296)==0
            disp(join(['progress:',num2str(cnt/296),' % ']))
        end                 
    end     
end




