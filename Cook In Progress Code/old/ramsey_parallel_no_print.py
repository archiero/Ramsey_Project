from setup import *
from pycuda_setup import *
from scipy.special import binom
import time

# np.random.seed(42)

# start = time.time()


# num_verts = 11
# ramsey = [3,10]
# num_colorings = 16


ramsey = np.asarray(ramsey)
ramsey.sort()
tot_verts = int(num_verts)
num_colors = len(ramsey)
num_verts_per_clique = ramsey
biggus_clickus = ramsey.max()  #If you did not just laugh, you need to get a life (of brian) - https://www.youtube.com/watch?v=zPGb4STRfKw
choose_table = np.array([[binom(n,k) for n in range(tot_verts+1)] for k in range(biggus_clickus+1)]).astype('uint64')
#NOTE: choose_table[k,n] gives binom(n,k).  This is strange, but works better later with the combinatorial numbering system.
tot_edges = int(choose_table[2,tot_verts])

num_edges_per_clique = np.array([choose_table[2,ram] for ram in ramsey])
num_cliques = np.array([choose_table[ram,tot_verts] for ram in ramsey])
num_cliques_cum = num_cliques.cumsum()
tot_cliques = int(num_cliques_cum[-1])

edge_idx_to_pair = np.array(list(it.combinations(range(tot_verts),2)))
edge_pair_to_idx = np.zeros([tot_verts,tot_verts])
for (idx,e) in enumerate(edge_idx_to_pair):
    edge_pair_to_idx[tuple(e)] = idx
edge_pair_to_idx += edge_pair_to_idx.T
np.fill_diagonal(edge_pair_to_idx,tot_edges)

colorings = np.random.randint(0, num_colors, size=[num_colorings, tot_edges], dtype = 'uint16')
#colorings[:,-1] = num_colors
problem_counts = np.zeros(num_colorings)

ramsey_gpu, ramsey = mtogpu(ramsey,'uint16')
choose_table_gpu, choose_table = mtogpu(choose_table,'uint32')
num_cliques_cum_gpu, num_cliques_cum = mtogpu(num_cliques_cum,'uint64')
edge_pair_to_idx_gpu, edge_pair_to_idx = mtogpu(edge_pair_to_idx,'uint16')
edge_idx_to_pair_gpu, edge_idx_to_pair = mtogpu(edge_idx_to_pair,'uint16')
colorings_gpu, colorings = mtogpu(colorings,'uint16')
problem_counts_gpu, problem_counts = mtogpu(problem_counts,'uint32')


kernel_code ="""
#include <stdio.h>

extern "C"
    #include <stdio.h>


    __device__ void edge_idx_to_pair_fcn(ushort *edge_idx_to_pair, ushort idx, ushort *v, ushort *w)
    {
        ushort start = 2*idx;
        *v = edge_idx_to_pair[start];
        *w = edge_idx_to_pair[start+1];    
    }


    __device__ void get_assignment(uint *choose_table, ushort *ramsey, ulong *num_cliques_cum, ushort *edge_idx_to_pair, ushort *edge_pair_to_idx, ushort *color, uint *clique_idx, ushort *n_verts, ushort *verts, ushort *n_edges, ushort *edges)
    {
        uint block_idx = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
        //uint thread_idx_loc = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        //uint thread_idx_glob = block_idx*blockDim.x*blockDim.y*blockDim.z + thread_idx_loc;

        ushort i, j, k, row, col, start;

        //First, determine which color we are.
        (*color) = 0;
        while(num_cliques_cum[(*color)] <= block_idx){(*color)++;}
        (*n_verts) = ramsey[(*color)];
        (*n_edges) = choose_table[2*(TOT_VERTS+1)+(*n_verts)];


        //Now, determine which clique_idx in that color we are.
        uint N = block_idx;
        if((*color) > 0)
        {
            N -= num_cliques_cum[(*color)-1];
        }
        (*clique_idx) = N;        


        //Now,  determine our list of vertices.
        for(i = 0; i < (*n_verts); i++)
        {
            row = (*n_verts) - i;
            start = row*(TOT_VERTS+1);
            col = 0;
            while(choose_table[start+col] <= N){
                col++;
            }
            col--;
            verts[row-1] = col;
            N -= choose_table[start+col];
        }

        //Now,  determine our list of edges.
        k = 0;
        ushort v, w;
        for(i = 0; i < (*n_verts); i++)
        {
            for(j = i+1; j < (*n_verts); j++)
            {
                v = verts[i];
                w = verts[j];
                edges[k] = edge_pair_to_idx[v*TOT_VERTS+w];
                k++;
            }
        }
    }


    __global__ void count_problems_gpu(ushort *colorings, uint *problem_counts, uint *choose_table, ushort *ramsey, ulong *num_cliques_cum, ushort *edge_idx_to_pair, ushort *edge_pair_to_idx)
    {
        //uint block_idx = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
        uint thread_idx_loc = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        //uint thread_idx_glob = block_idx*blockDim.x*blockDim.y*blockDim.z + thread_idx_loc;

        uint i, start;
        
        __shared__ ushort color, n_verts, n_edges;
        __shared__ uint clique_idx;

        __shared__ ushort verts[MAX_VERTS];
        for(i = 0; i < MAX_VERTS; i++){verts[i] = TOT_VERTS;}
        
        __shared__ ushort edges[MAX_EDGES];
        for(i = 0; i < MAX_EDGES; i++){edges[i] = TOT_EDGES;}
        
        if(thread_idx_loc == 0)
        {
            get_assignment(choose_table, ramsey, num_cliques_cum, edge_idx_to_pair, edge_pair_to_idx, &color, &clique_idx, &n_verts, verts, &n_edges, edges);
        }
        __syncthreads();
        
        start = thread_idx_loc * TOT_EDGES;
        i = 0;
        while((i < n_edges) && (colorings[start+edges[i]] == color)){i++;}
        
        if(i >= n_edges)
        {
            atomicAdd(&problem_counts[thread_idx_loc], 1);
        }
    }
"""


regex_fixing_dictionary = {"TOT_VERTS":tot_verts, "TOT_EDGES":tot_edges, "TOT_CLIQUES": tot_cliques
                          ,"MAX_VERTS":biggus_clickus, "MAX_EDGES":choose_table[2,biggus_clickus]
                          ,"PRINT_ID":99999999999999}
for key, val in regex_fixing_dictionary.items():
    kernel_code = kernel_code.replace(str(key),str(val))

mod = SourceModule(kernel_code)
count_problems_gpu = mod.get_function("count_problems_gpu")

block_dims = (num_colorings,1,1)
grid_dims = (tot_cliques,1,1)

def count_problems(colorings,printout=False):
    colorings_gpu.set(colorings.ravel())    
    count_problems_gpu(colorings_gpu, problem_counts_gpu, choose_table_gpu, ramsey_gpu, num_cliques_cum_gpu, edge_idx_to_pair_gpu, edge_pair_to_idx_gpu, block=block_dims, grid=grid_dims, shared=0)
    return problem_counts_gpu.get()


# problem_counts = count_problems(colorings)
# print(problem_counts)
# #print(colorings)


# end = time.time()
# print("time = %f"%(end-start))