def AddGlobalNodes(graph, localgraph, nodeflows):
    '''Add one hop nodes in global graph to the local graph
    Parameters
    ----------
    graph : the whole graph
    subgraph : the subgraph of local client
    nodeflows : the nodeflow list get from sampling  *** one hop nodeflow sampling from gobal graph 
    Return
    ----------
    subgraph : the subgraph contains one hop nodes in global
    '''
    nodes_in_global = []
    for nf in nodeflows:
        # list to set is used to remove repeated nodes
        nodes_in_global = nodes_in_global + nf.layer_parent_nid(0).tolist() + nf.layer_parent_nid(1).tolist()
        nodes_in_global_tmp = set(nodes_in_global)
        nodes_in_global = list(nodes_in_global_tmp)
    nodes_in_local = localgraph.parent_nid.tolist()
    nodes_return = nodes_in_local + nodes_in_global
    nodes_return_tmp = set(node_return)
    nodes_return = list(nodes_return_tmp)
    subgraph = graph.subgraph(nodes_return)
    subgraph.copy_from_parent()
    return subgraph