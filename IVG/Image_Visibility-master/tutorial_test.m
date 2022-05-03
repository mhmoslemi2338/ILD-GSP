%% TUTORIAL: Lena Analysis 
clc

addpath ../../'Graph wavelet implement'/matlabTools/gspbox/
gsp_start
% gsp_install




%% extract the (horizontal) image visibility graph (IHVG)
clc
close all
x0=10;
y0=10;
width=1000;
height=1000;

names={'Test/E','Test/F','Test/M','Test/H','Test/GG'};

for name=names
    name=cell2mat(name);

I=uint8(imread(join([name,'.png'])));
% figure;imagesc(I);colormap('gray');axis off

% I=I(1:2:32,1:2:32);
Edge_list=imageVisibilityGraph(I,'horizontal',false);
G = graph(Edge_list(:,1),Edge_list(:,2));


G_edges=table2array(G.Edges);
A1=zeros(32^2,32^2);
for i=[1:length(G_edges)]
    nodes=G_edges(i,:);
    A1(nodes(1),nodes(2))=1;
    A1(nodes(2),nodes(1))=1;
end


G1=(gsp_graph(sparse(A1),gsp_full_connected(1024).coords));
figure;set(gcf,'position',[x0,y0,width,height]); gsp_plot_graph(G1);
saveas(gcf,join([name,'_full_connected.jpg']))
G1=(gsp_graph(sparse(A1),gsp_2dgrid(32).coords));
figure;set(gcf,'position',[x0,y0,width,height]); gsp_plot_graph(G1);
saveas(gcf,join([name,'_2dgrid.jpg']))
figure;set(gcf,'position',[x0,y0,width,height]); plot(G);
saveas(gcf,join([name,'_graph.jpg']))
close all


% %%%%%%%%%%%%%%%%%%
% apply IHVG k-filter (degree filter) to the Lena image

% degree sequence from the IHVG
Deg_seq = degree(G);

% Degree distribution P(k)
Pk=hist(Deg_seq,1:1:max(Deg_seq))./(size(I,1)*size(I,2));

% Pixel intensity distribution from original image
Pi=hist(double(I(:)),1:1:max(double(I(:))))./(size(I,1)*size(I,2));


%filter construction
if length(Deg_seq)==1023
    I2 = reshape([Deg_seq; 0],32,32);
else
    I2 = reshape(Deg_seq,32,32);
end


I2 = uint8(full(I2'));


figure
subplot(2,2,1);imagesc(I);colormap('gray');axis off;title('Lena original');
subplot(2,2,2);plot(1:1:max(double(I(:))),Pi);xlabel('Pixel Intensity');ylabel('Frequency'); xlim([1,256])
subplot(2,2,3);imagesc(I2);colormap('gray');axis off;title('k-filter');
subplot(2,2,4);plot(1:1:max(Deg_seq),Pk);xlabel('Pixel Intensity');ylabel('Frequency');
saveas(gcf,join([name,'_k-filter.jpg']))
% %%%%%%%%%%%%%%%%%%
% number of pixels/nodes in the graphs 
% close all
clc

N=size(I,1).^2;
Iseq={I};
Pk=cell(1,numel(Iseq));
Z=cell(1,numel(Iseq));

% Set a maximum degree 
maxK=max(Deg_seq);
for i=1:numel(Iseq)
Edge_list=imageVisibilityGraph(Iseq{i},'horizontal',true);
Deg_seq = hist(Edge_list(:,1),1:N)+hist(Edge_list(:,2),1:N);
Pk{i}=hist(Deg_seq,1:1:maxK)./N;
Z{i}=visibilityPatches(Iseq{i},1,'horizontal');
end

figure;
subplot(2,1,1);
plot(1:1:maxK,Pk{1}); hold on;
plot(1:1:maxK,Pk{1},'Marker','.','Markersize',13,'Linestyle','none');
xlim([0,maxK])
xlabel('k');ylabel('Frequency');
subplot(2,1,2);
stem((Z{1}),'.');
set(gca,'yscal','log');
xlim([1,256])
xlabel('patch id');ylabel('Frequency (log10)');
saveas(gcf,join([name,'_features.jpg']))
close all


end