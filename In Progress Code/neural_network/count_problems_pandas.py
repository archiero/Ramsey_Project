from setup import *
from scipy.special import binom as choose

Ramsey = ramsey.copy()
num_colors = len(Ramsey)
try:
    num_vertices = num_verts
except NameError:
    pass
Colors = np.arange(num_colors)
Vertices = np.arange(num_vertices)
Edges = list(it.combinations(Vertices,2))

#We're saving the graph as a 1-dimensional array and this dictionary allows us to go from coloring to array easily
#reverse lookup for edges below.  Eg if slot 3 above contains edge (2,5), the the dict below has entry (2,5):3
Edges_idx = dict((edge, idx) for idx,edge in enumerate(Edges))
num_edges = int(len(Edges))

threads_per_block = 512
vertices_per_clique = Ramsey
edges_per_clique = np.array([choose(v,2) for v in vertices_per_clique]).astype('uint32')
cliques_per_color = np.asarray([choose(num_vertices,v) for v in vertices_per_clique])
blocks_per_color = np.ceil(cliques_per_color / threads_per_block).astype('uint32')
num_blocks = blocks_per_color.sum()
cliques_per_block = np.ceil(cliques_per_color / blocks_per_color).astype('uint32')

#The objects below tells each block which color and how many cliques/edges it will monitor.
#Note each vector is repetitive.  If color 0 gets 7 blocks, the first 7 entries will be the same
block_color = np.repeat(Colors,blocks_per_color).astype('uint32')
block_num_cliques = np.repeat(cliques_per_block,blocks_per_color).astype('uint32')
block_edges_per_clique = np.repeat(edges_per_clique,blocks_per_color).astype('uint32')

#The object below assigns each block to a list of cliques.  For simplicity while
#we construct a single matrix with a lot of unused entries.
#For example, edges_per_clique is different for blocks monitoring different colors.
#We could make a more complex structure that handles this (see old stable version)
#but that makes it harder to pass to the GPU.  Instead, we simply fill all "invalid"
#unused entries with the placeholder "num_edges", which is 1 more the the largest
#legal edge_idx (because Python indexes [0,1,2,...,num_edges-1]).
#When we color later, we color these slots with the placeholder num_colors.
assign_Blocks_to_Cliques = np.full([int(num_blocks),int(cliques_per_block.max()),int(edges_per_clique.max())],
                                   fill_value=num_edges, dtype='uint32')

#Counters that that tracks the next open block and thread on each block
next_open_block = 0
next_open_thread = np.zeros(num_blocks,dtype='int')
for color, clique_size in enumerate(Ramsey):
    #Creates a generator to produce all cliques (the list of vertices).
    Cliques = it.combinations(Vertices,clique_size)

    #Makes the vector [0,1,2,...,num_blocks-1,0,1,2,...,num_blocks-1,....] of length num_cliques
    assign_Cliques_to_Blocks = np.arange(cliques_per_color[color]) % blocks_per_color[color]
    assign_Cliques_to_Blocks = np.asarray(assign_Cliques_to_Blocks).astype('int32')
    #randomizes assignment, but maintains clique counts
    np.random.shuffle(assign_Cliques_to_Blocks)
    #Starts at next open block
    assign_Cliques_to_Blocks += next_open_block
    
    for clique_Vertices, block in zip(Cliques,assign_Cliques_to_Blocks):
        #Gets the list of edges in this clique
        clique_Edges = list(it.combinations(clique_Vertices,2))
        #Converts it to edge_idx
        clique_Edges_idx = [Edges_idx[edge] for edge in clique_Edges]
        #Writes it to the correct block and next open thread on that block
        assign_Blocks_to_Cliques[block,next_open_thread[block],:edges_per_clique[color]] = clique_Edges_idx
        next_open_thread[block] += 1
    next_open_block += blocks_per_color[color]

#The code below setups the serial version of this algorithm in pandas on the CPU.
#It is much slower than the gpu version, but can be used in the absence of a GPU
#and to verify that the algorithms give the same answers.
#We msut create the "comparison" array.  This is a bit complicated.
#We will discuss 2 arrays: compare and the coloring array.
#First, recall that num_colors is one larger than the biggest legal color since
#Python indexes [0,1,...,num_colors-1]
#Now, fix a block and let c = block_color[block].
#Consider the [block, clique, edge] entry of compare.  It equals:
#c IF edge < num_edges_per_clique for that block
#num_colors IF clique >= num_edges_per_clique for that block
#Why?  In general there are extra rows and columns not associated to a valid edge.
#When we color the graph later, they are filled with num_colors.
#We do NOT want the "space fillers" to affect problem count.
#In the extra rows, we see [c,c,...,c,num_colors,num_colors,...,num_colors] in compare
#But in the coloring array, all entries will equal num_colors.
#Thus, it is NOT counted as a problem because the first several slots disagree.
#Thus, these extra rows can NEVER counts as problems cliques, as desired.
#Now, consider the extra columns.  All entries will equal num_colors.  This is true
#for BOTH compare AND the coloring array.  Thus, a row counts as a problem
#if and only if the first num_edges_per_clique "valid" entries also match.
#Thus the extra columns do NOT alter the "problem status" for valid rows, as desired.

compare = np.full_like(assign_Blocks_to_Cliques, fill_value=num_colors)
print(assign_Blocks_to_Cliques.shape)
for block in range(num_blocks):
    compare[block,:,:block_edges_per_clique[block]] = block_color[block]

def count_problems(coloring, printout=False):
    coloring = np.ravel(coloring)
    X = coloring[assign_Blocks_to_Cliques]
    Y = (X == compare)
    Problems = np.all(Y,axis=-1)
    if printout == True:
        print_problems(Problems)
    return Problems.sum().astype('int')

def print_problems(Problems):
    Z = pd.DataFrame(Problems.astype('uint32'))
    Z.insert(0,'problems',Z.sum(axis=1))
    Z.insert(0,'color',block_color)
    Z = Z.T
    problem_idx = np.any(Z,axis=-1)
    problem_idx[:2] = True
    display(Z.ix[problem_idx,:])

def print_status():
    now = time.time()
    elapsed = now - start
    print("%d steps done in %s.  Best coloring so far was step %d with %d problems.  Time now %s."
               %(step,str(elapsed).split('.')[0],best_step,problems_best,str(now).split('.')[0]))

#######This is dumbsey as a function. It's a very slow but highly intuitive way to count problems
#edge_idx_to_pair = np.array(list(it.combinations(range(num_vertices),2)))
#edge_pair_to_idx = np.zeros([num_vertices, num_vertices])
#for (idx,e) in enumerate(edge_idx_to_pair):
#    edge_pair_to_idx[tuple(e)] = idx
#edge_pair_to_idx += edge_pair_to_idx.T
#np.fill_diagonal(edge_pair_to_idx, choose(num_vertices, 2))
#
#def dumbsey(coloring):
#    count = 0
#    for i, ram in enumerate(ramsey):
#        cliques = it.combinations(range(num_vertices),ram)
#       for j, v in enumerate(cliques):
#            edges = list(it.combinations(v,2))
#            c = []
#            for k, edge in enumerate(edges):
#                edge_idx = int(edge_pair_to_idx[tuple(edge)])
#                c.append(coloring[edge_idx])
#            c = np.asarray(c)
#            count += np.all(c == i)
#    return(count)