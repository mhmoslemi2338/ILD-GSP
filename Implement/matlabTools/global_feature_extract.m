
function global_feature=global_feature_extract(G)
Deg_seq = degree(G);
N=1024;
G_edges=table2array(G.Edges);
A1=zeros(N,N);
for i=[1:length(G_edges)]
    nodes=G_edges(i,:);
    A1(nodes(1),nodes(2))=1;
end
A1 = (A1 + A1')- diag(diag(A1));


% Degree distribution P(k)
Pk=hist(Deg_seq,1:1:max(Deg_seq))./N; 
Pk_avg = mean(Pk(Pk > 0)); 
Pk_median = median(Pk(Pk > 0));
Pk_max=max(Pk(Pk > 0));
Pk_min=min(Pk(Pk > 0));
% Pk_wb=wblfit(Pk(Pk>0));


%The local clustering coefficient of each node
cn = diag(A1*triu(A1)*A1); %Number of triangles for each node
c = zeros(size(Deg_seq));
c(Deg_seq > 1) = 2 * cn(Deg_seq > 1) ./ (Deg_seq(Deg_seq > 1).*(Deg_seq(Deg_seq > 1) - 1)); 
C_avg = mean(c(Deg_seq > 1)); 
C_median = median(c(Deg_seq > 1));
C_max=max(c(Deg_seq > 1));
C_min=min(c(Deg_seq > 1));
C_wb=wblfit(c(c>0));

% average nearest-neighbors degree
Knn=[];
for i=[1:max(max(table2array(G.Edges)))]
    neighbours=[];
    for j=neighbors(G,i)
        neighbours=[neighbours degree(G,j)];
        Knn=[Knn round(mean(neighbours))];
    end
end
Knn=Knn/N;
Knn_avg = mean(Knn); 
Knn_median = median(Knn);
Knn_max=max(Knn);
Knn_min=min(Knn);
% Knn_wb=wblfit(Knn(Knn>0));

global_feature=[Pk_avg Pk_median Pk_max Pk_min C_avg C_median C_max C_min Knn_avg Knn_median Knn_max Knn_min];

end

