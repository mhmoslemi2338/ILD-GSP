


clear all
clc
close all
cnt=0;

addpath matlabTools/


% prepare folders
if ~ isdir('features')
    mkdir('features');
end
if ~ isdir('features/Train_ILD')
    mkdir('features/Train_ILD');
end
if ~ isdir('features/Test_Talisman')
    mkdir('features/Test_Talisman');
end


data_type={'Train_ILD','Test_Talisman'};
for idx=1:2
    typee=data_type{idx};
    
    lables=ls(join(['wavelet_mat/',typee]));
    lables=(split(strip(lables)));
    path={};
    for k=1:length(lables)
        ll=lables{k};
        path=[path; join(['wavelet_mat/',typee,'/',ll])];
        if ~ isdir(join(['features/',typee,'/',ll]))
            mkdir(join(['features/',typee,'/',ll]))
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
end


