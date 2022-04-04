
function [G1,G2]=define_graph_gspbox(graph_seize)

    g1 = imageGraph([graph_seize graph_seize],4);
    g1_edges=table2array(g1.Edges);
    g1_edges(:,3) = [];
    A1=zeros(graph_seize^2,graph_seize^2);
    for i=[1:length(g1_edges)]
        nodes=g1_edges(i,:);
        A1(nodes(1),nodes(2))=1;
    end
    A1 = A1 - diag(diag(A1));
    A1 = (A1 + A1');
    
    
    conn=[1 0 1; 0 1 0; 1 0 1;];
    g2 = imageGraph([graph_seize graph_seize],conn);
    g2_edges=table2array(g2.Edges);
    g2_edges(:,3) = [];
    A2=zeros(graph_seize^2,graph_seize^2);
    for i=[1:length(g2_edges)]
        nodes=g2_edges(i,:);
        A2(nodes(1),nodes(2))=1;
    end
    A2 = A2 - diag(diag(A2));
    A2 = (A2 + A2');
    
    G = gsp_2dgrid(graph_seize);
    G1=gsp_estimate_lmax(gsp_graph(sparse(A1),G.coords));
    G2=gsp_estimate_lmax(gsp_graph(sparse(A2),G.coords));


end
