# 
'''
nodeflows = dgl.contrib.sampling.NeighborSampler(...):
nodeflows is the list of nodeflows from sampling
'''
# G = <V, E> = {G1,G2,...} G1 is local graph, other graphs from other clients

def DelNonLocalEdges(graph, subgraph, nodeflows):
    ''' Delete the edges not in the local graph.
    
    Parameters
    ----------
    graph : the whole graph
    subgraph : the subgraph of local client
    nodeflows : the nodeflow list get from sampling
    
    Return
    ----------
    Subgraph : the list of processed nodeflows in DGLGraph format
    '''
    subgraphs = []
    seed_nid = []

    for nf in nodeflows:
        # 取第二层以及第三层的节点在总图上的id
        middle_layer = nf.layer_parent_nid(1)
        bottom_layer = nf.layer_parent_nid(0)        

        # 建立子图
        edges_list = []
        for blocks in range(nf.num_blocks()-1):
            edges_list = edges_list + nf.block_parent_eid(blocks).tolist
        tmp_set = set(edges_list)
        tmp_list = list(tmp_set)
        tmp_graph = graph.edge_subgraph(tmp_list)
        
        # 判断并删边
        for node_m in middle_layer:
            sub_node_m = subgraph.map_to_subgraph([node_m])
            if subgraph.has_node(sub_node_m.tolist[0]):
                for node_b in bottom_layer:
                    sub_node_b = subgraph.map_to_subgraph([node_b])
                    if subgraph.has_node(sub_node_b.tolist[0])==False:
                        tmp_graph.remove_edges(graph.edge_id(node_m, node_b))
                        tmp_graph.remove_edges(graph.edge_id(node_b, node_m))
        seed_nid = seed_nid + nf.layer_parent_nid(-1)[0]
        subgraphs = subgraph + tmp_graph
                        
    return subgraphs,seed_nid

def FullSamplingForList(graphs):
    ''' Run Full Sampling algorithm for the graph list
    
    Parameters
    ----------
    garph : the list of processed nodeflows in DGLGraph format
    
    Return 
    ----------
    nfs : the list of nodeflows   
    '''
    nfs = []
    for graph in graphs:
        test_batch_sampling_method = np.array([])
        for layer in range(2):
            for nodes in range(graphs.number_of_nodes()):
                test_batch_sampling_method = np.append(test_batch_sampling_method, 10000)
        nf = dgl.contrib.sampling.NeighborSampler(graph, 10000,
                                            expand_factor = test_batch_sampling_method,
                                            neighbor_type='in',
                                            shuffle=True,
                                            num_workers=32,
                                            num_hops=3,
                                            seed_nodes=None)
        nfs = nfs + nf
    return nfs