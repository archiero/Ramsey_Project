from setup import *
from pycuda_setup import *
from scipy.special import binom

# np.random.seed(42)

# num_verts = 20
# ramsey = [3,10]
# num_colorings = 1






ramsey = np.asarray(ramsey)
ramsey.sort()
tot_verts = int(num_verts)
num_clrs = len(ramsey)
num_verts_per_clique = ramsey
biggus_clickus = ramsey.max()  #If you did not just laugh, you need to get a life (of brian) - https://www.youtube.com/watch?v=zPGb4STRfKw
choose_table = np.array([[binom(n,k) for n in range(tot_verts+1)] for k in range(biggus_clickus+1)]).astype('uint64')
#NOTE: choose_table[k,n] gives binom(n,k).  This is strange, but works better later with the combinatorial numbering system.
tot_edges = int(choose_table[2,tot_verts])


num_edges_per_clique = np.array([choose_table[2,ram] for ram in ramsey])
num_cliques = np.array([choose_table[ram,tot_verts] for ram in ramsey])
num_cliques_cum = num_cliques.cumsum()
tot_cliques = int(num_cliques_cum[-1])

colorings = np.random.randint(0, num_clrs, size=[num_colorings, tot_edges+1], dtype = 'uint16')
colorings[:,-1] = num_clrs

####
problems = np.zeros([num_colorings,tot_cliques])
####

edge_idx_to_pair = np.array(list(it.combinations(range(tot_verts),2)))
edge_pair_to_idx = np.zeros([tot_verts,tot_verts])
for (idx,e) in enumerate(edge_idx_to_pair):
    edge_pair_to_idx[tuple(e)] = idx
edge_pair_to_idx += edge_pair_to_idx.T
np.fill_diagonal(edge_pair_to_idx,tot_edges)



ramsey_gpu, ramsey = mtogpu(ramsey,'uint16')
choose_table_gpu, choose_table = mtogpu(choose_table,'uint32')
num_cliques_cum_gpu, num_cliques_cum = mtogpu(num_cliques_cum,'uint64')
edge_pair_to_idx_gpu, edge_pair_to_idx = mtogpu(edge_pair_to_idx,'uint16')
edge_idx_to_pair_gpu, edge_idx_to_pair = mtogpu(edge_idx_to_pair,'uint16')
colorings_gpu, colorings = mtogpu(colorings,'uint16')


####
problems_gpu, problems = mtogpu(problems,'uint16')
####



#Optionally, return the clique list to python.  TURN ME OFF for large jobs or I will eat all your brains (RAM).
# clique_list = np.ones([tot_cliques,biggus_clickus]) * tot_verts
# clique_list_gpu, clique_list = mtogpu(clique_list,'uint16')

# edge_list = np.ones([tot_cliques,choose_table[2,biggus_clickus]]) * tot_edges
# edge_list_gpu, edge_list = mtogpu(edge_list,'uint16')



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


    __device__ void get_assignment(uint *choose_table, ushort *ramsey, ulong *num_cliques_cum, ushort *edge_idx_to_pair, ushort *edge_pair_to_idx, ushort *clr, uint *clique_idx, ushort *n_verts, ushort *verts, ushort *n_edges, ushort *edges)
    {
        ushort i, j, k, row, col, start;
        uint block_idx = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;

        //First, determine which color we are.
        (*clr) = 0;
        while(num_cliques_cum[(*clr)] <= block_idx){(*clr)++;}
        (*n_verts) = ramsey[(*clr)];
        (*n_edges) = choose_table[2*(TOT_VERTS+1)+(*n_verts)];


        //Now, determine which clique_idx in that color we are.
        uint N = block_idx;
        if((*clr) > 0)
        {
            N -= num_cliques_cum[(*clr)-1];
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
        //printf("block_idx = %5u, color = %1u, n_verts = %2u, n_edges = %3u, clique_idx = %5u\\n", block_idx, (*clr), (*n_verts), (*n_edges), (*clique_idx));


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
                if(block_idx == PRINT_ID)
                {
                    ushort r,s;
                    edge_idx_to_pair_fcn(edge_idx_to_pair, edges[k], &r, &s);
                    printf("edge (%u,%u) -> edge_idx %u -> (%u,%u)\\n",v,w,edges[k],r,s);                    
                }
                k++;
            }
        }
    }


    __global__ void count_problems_gpu(ushort *colorings, ushort *problems, uint *choose_table, ushort *ramsey, ulong *num_cliques_cum, ushort *edge_idx_to_pair, ushort *edge_pair_to_idx)//, ushort *clique_list, ushort *edge_list)
    {
        uint i, start;
        uint block_idx = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
        uint thread_idx_loc = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        uint thread_idx_glob = block_idx*blockDim.x*blockDim.y*blockDim.z + thread_idx_loc;
        
        __shared__ ushort clr, n_verts, n_edges;
        __shared__ uint clique_idx;

        __shared__ ushort verts[MAX_VERTS];
        for(i = 0; i < MAX_VERTS; i++){verts[i] = TOT_VERTS;}
        
        __shared__ ushort edges[MAX_EDGES];
        for(i = 0; i < MAX_EDGES; i++){edges[i] = TOT_EDGES;}
        
        if(thread_idx_loc == 0)
        {
            get_assignment(choose_table, ramsey, num_cliques_cum, edge_idx_to_pair, edge_pair_to_idx, &clr, &clique_idx, &n_verts, verts, &n_edges, edges);
        
            //Everything beyone this point is optional
            //printf("block_idx = %5u, color = %1u, n_verts = %2u, n_edges = %3u, clique_idx = %5u\\n", block_idx, clr, n_verts, n_edges, clique_idx);

            //print from specified block
            if(block_idx == PRINT_ID)
            {
                for(i = 0; i < MAX_VERTS; i++)
                {
                    printf("%u ",verts[i]);
                }
                printf("\\n");

                ushort v, w;
                for(i = 0; i < MAX_EDGES; i++)
                {
                    edge_idx_to_pair_fcn(edge_idx_to_pair, edges[i], &v, &w);
                    printf("edge_idx %u -> (%u,%u)", edges[i], v, w);
                        printf("\\n");
                }
            }        

            //Optionally, return the clique list to python.  TURN ME OFF for large jobs or I will eat all your brains (RAM).
            /*
            start = block_idx*MAX_VERTS;
            for(i = 0; i < n_verts; i++)
            {
                clique_list[start+i] = verts[i];
            }
            
            start = block_idx*MAX_EDGES;
            for(i = 0; i < n_edges; i++)
            {
                edge_list[start+i] = edges[i];                
            }
            */

        }
        __syncthreads();
       
        //printf("(block_idx,thread_idx) = (%5u,%3u), color = %1u, n_verts = %2u, n_edges = %3u, clique_idx = %5u\\n", block_idx, thread_idx_loc, clr, n_verts, n_edges, clique_idx);
       
        start = thread_idx_loc * TOT_EDGES;
        i = 0;
        while((i < n_edges) && (colorings[start+edges[i]] == clr)){i++;}
       
        uint prob_idx = TOT_CLIQUES * thread_idx_loc + block_idx;
        if(i >= n_edges)
        {
            problems[prob_idx] = 1;
        }
        else
        {
            problems[prob_idx] = 0;
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

def count_problems(colorings, printout=False):
    colorings_gpu.set(colorings.ravel())    
    count_problems_gpu(colorings_gpu, problems_gpu, choose_table_gpu, ramsey_gpu, num_cliques_cum_gpu, edge_idx_to_pair_gpu, edge_pair_to_idx_gpu, block=block_dims, grid=grid_dims, shared=0)

#    count_problems_gpu(colorings_gpu, problems_gpu, choose_table_gpu, ramsey_gpu, num_cliques_cum_gpu, edge_idx_to_pair_gpu, edge_pair_to_idx_gpu, clique_list_gpu, edge_list_gpu, block=block_dims, grid=grid_dims, shared=0)
    # print(edge_list)
    problems = problems_gpu.get().reshape([num_colorings,tot_cliques])
    #problems = problems.T

    if printout == True:
        clique_list = clique_list_gpu.get().reshape([tot_cliques,-1])
        edge_list = edge_list_gpu.get().reshape([tot_cliques,-1])
        for i,(c,e) in enumerate(zip(clique_list,edge_list)):
            print("%3u  %s  %s  %s  %u"%(i, c, e, colorings[0,e], problems[0,i])) 

    return problems.sum(axis=1), problems






#print(colorings)
#num_problems, problems = count_problems(colorings,printout=False)
# print(clique_list.shape)
# print(problems.shape)
#print(num_problems)
# print(problems)
# print(problems.shape)
# print(tot_cliques)