


clear all
clc
close all
cnt=0;

addpath matlabTools/






dest_dir='wavelet_features';

lables=ls('/Users/mohammad/Desktop/Bsc prj/Graph wavelet implement/data/Train_ILD');
lables=split(strip(lables));

% prepare folders
if ~ isdir(dest_dir)
    mkdir(dest_dir);
end


path={};
for k=1:length(lables)
    ll=lables{k};
    path=[path; join(['/Users/mohammad/Desktop/Bsc prj/Graph wavelet implement/data/Train_ILD/',ll])];
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

        if length(strfind(filename,'level1'))==1

            filename=filename(1:end-5);
            feature_vector=extract_features(filename);
            savepath=replace(filename,'_level','_feature.mat');
            savepath=replace(savepath,'wavelet_mat','features');
            save(savepath,'feature_vector'); %save wavelet response
            cnt=cnt+1;
            %print progress
            if mod(cnt,485)==0
                disp(join(['progress:',num2str(cnt/485),' % ']))
            end

        end
    end     
end



