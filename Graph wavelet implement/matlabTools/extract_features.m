



function vector=extract_features(input_file)

    mystep=4;
    vector=[];
    for level=1:2;
    load(join([input_file,num2str(level),'.mat']));
    mat=wavelet_level_sq;
    
    
    for idx=1:4
        mymat=mat(:,:,idx);
        [m,n]=size(mymat);
        F1=[];
        F2=[];
        F3=[];
        
        for i=1:m-mystep+1
            for j=1:n-mystep+1
                 window=mymat(i:i+mystep-1,j:j+mystep-1);
                 singularValue=svd(window);
                 F1=[F1; max(singularValue)];
                 F2=[F2; mean(singularValue)];
                 F3=[F3; median(singularValue)];
            end
        end
    
        F1_param=wblfit(F1);
        F2_param=wblfit(F2);
        F3_param=wblfit(F3);
    
    vector=[vector F1_param F2_param F3_param];
    
    end
    end

end