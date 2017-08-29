#from setup import *
#from scipy.special import binom

#num_verts = 20
#ramsey = [3,10]
#num_colorings = 1

num_verts = 5
ramsey = [3,3] 
ramsey = np.asarray(ramsey)
ramsey.sort()
colorings = np.array([[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2]])
tot_verts = num_verts
tot_edges = binom(tot_verts,2)

edge_idx_to_pair = np.array(list(it.combinations(range(tot_verts),2)))
edge_pair_to_idx = np.zeros([tot_verts,tot_verts])
for (idx,e) in enumerate(edge_idx_to_pair):
    edge_pair_to_idx[tuple(e)] = idx
edge_pair_to_idx += edge_pair_to_idx.T
np.fill_diagonal(edge_pair_to_idx,tot_edges)


prob_count = 0
for i, ram in enumerate(ramsey):
    cliques = it.combinations(range(num_verts),ram)    
    for j,v in enumerate(cliques):        
        edges = list(it.combinations(v,2))
        c = np.zeros(len(edges))
        for k,edge in enumerate(edges):
            edge_idx = np.int(edge_pair_to_idx[tuple(edge)])
            #c.append(colorings[0,edge_idx])
            c[k] = colorings[0,edge_idx]
        if np.all(c == i) == True:
            print("clr%u  clique%u  verts%s  edges%s  colors%s "%(i,j,v,edge,c))
            prob_count += 1
print("number of problems are %u"%(prob_count))
