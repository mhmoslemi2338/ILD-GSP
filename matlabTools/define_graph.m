

function [Gs ,N ,Ln_bpt ,Colorednodes ,beta_dist] = define_graph(Data , theta , max_level)
    
% Parameters ####
    edgemap =1; % uses edge-map if 1, use regular 8-connected graph otherwise
    m=length(Data);
    N = zeros(max_level,1); %size of bipartite graphs at each level

    
% Graphs ####
    [bptG Colorednodes, beta_dist loc] = image_graphs_multi(Data,max_level,edgemap);  % image graphs
    
% Compute Normalized Laplacian Matrices for Each Bpt graph ####
    Ln_bpt = cell(max_level,theta);
    for level = 1:max_level
        N(level) = length(bptG{level,1});
        for i = 1:theta
            d1 = sum(bptG{level,i},2);
            d1(d1 == 0) = 1; % for isolated nodes
            d1_inv = d1.^(-0.5);
            D1_inv = spdiags(d1_inv, 0, N(level), N(level));
            An = D1_inv*bptG{level,i}*D1_inv;
            An = 0.5*(An + An');
            Ln_bpt{level,i} = speye(N(level)) - An;
        end
    end
    
% GSPbox implementation ####
    G1_L1 = gsp_estimate_lmax(gsp_graph(sparse(bptG{1,1}),gsp_2dgrid(m).coords));
    G2_L1 = gsp_estimate_lmax(gsp_graph(sparse(bptG{1,2}),gsp_2dgrid(m).coords));
    G1_L2 = gsp_estimate_lmax(gsp_graph(sparse(bptG{2,1}),gsp_2dgrid(m).coords));
    G2_L2 = gsp_estimate_lmax(gsp_graph(sparse(bptG{2,2}),gsp_2dgrid(m).coords));
    
    G1_L1.LN=Ln_bpt{1,1};
    G2_L1.LN=Ln_bpt{1,2};
    G1_L2.LN=Ln_bpt{2,1};
    G2_L2.LN=Ln_bpt{2,2};

    Gs={'G1_L1',G1_L1;
        'G2_L1',G2_L1;
        'G1_L2',G1_L2;
        'G2_L2',G2_L2};

end