


function plot_wavelet_response(Sf,graph_size,level)



g1 = imageGraph([graph_size graph_size],8);
g1_edges=table2array(g1.Edges);
g1_edges(:,3) = [];
A1=zeros(graph_size^2,graph_size^2);
for i=[1:length(g1_edges)]
    nodes=g1_edges(i,:);
    A1(nodes(1),nodes(2))=1;
end
A1 = A1 - diag(diag(A1));
A1 = (A1 + A1');
G1=gsp_estimate_lmax(gsp_graph(sparse(A1),gsp_2dgrid(graph_size).coords));





figure1=figure('Position', [100, 100, 1024, 1200]);
param_plot.cp = [0.1223, -0.3828, 12.3666];
subplot(221)
gsp_plot_signal(G1,Sf(:,1), param_plot);
axis square; title(['Wavelet 1 , Level ',num2str(level)]); c_scale = 4;
caxis([mean(Sf(:,1)) - c_scale*std(Sf(:,1));, mean(Sf(:,1)) + c_scale*std(Sf(:,1));]);
subplot(222)
gsp_plot_signal(G1,Sf(:,2), param_plot);
axis square; title(['Wavelet 2 , Level ',num2str(level)]);
caxis([mean(Sf(:,2)) - c_scale*std(Sf(:,2));, mean(Sf(:,2)) + c_scale*std(Sf(:,2));]);
subplot(223)
gsp_plot_signal(G1,Sf(:,3), param_plot);
axis square; title(['Wavelet 3 , Level ',num2str(level)]);
caxis([mean(Sf(:,3)) - c_scale*std(Sf(:,3));, mean(Sf(:,3)) + c_scale*std(Sf(:,3));]);
subplot(224)
gsp_plot_signal(G1,Sf(:,4), param_plot);
axis square; title(['Wavelet 4 , Level ',num2str(level)]);
caxis([mean(Sf(:,4)) - c_scale*std(Sf(:,4));, mean(Sf(:,4)) + c_scale*std(Sf(:,4));]);

end