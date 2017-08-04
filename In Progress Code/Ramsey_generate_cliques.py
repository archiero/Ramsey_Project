from setup import *
#from __future__ import division
import random

def main_random(Ramsey, num_vertices, num_steps, beta=2):
    start_time = datetime.datetime.now()
    print()
    Ramsey = np.array(Ramsey)
    def choose(k,n):
        try: 
            return int(math.factorial(n)/math.factorial(k)/math.factorial(n-k))
        except:
            return 0
    #The theory for this function comes from the Lesser Paper. Theorem 7 there provides very useful lower-bounds 
    #for the number of red edges found 
    def min_red_edges(Ramsey, num_vertices):
        sorted_Ramsey = np.sort(Ramsey)
        k = sorted_Ramsey[1]
        if len(sorted_Ramsey) != 2 or sorted_Ramsey[0] != 3:
            return(0)
        elif num_vertices <=2*k:
            return(num_vertices-k)
        elif num_vertices <= 5*k/2:
            return(3*num_vertices - 5*k)
        else:
            return(5*num_vertices - 10*k)
    #takes an array and returns the PyCUDA tag and numpy version of said array with the identified data type    
    def mtogpu(arr, dtype=None):
        arr = np.asarray(arr)
        if dtype is not None:
            arr = arr.astype(dtype)    
        arr_gpu = gpuarray.to_gpu(arr.ravel())
        return arr, arr_gpu
    #the color "red" is always assumed to 0 and the smallest of the clique sizes that we're looking for
    min_red = min_red_edges(Ramsey, num_vertices)
    red = 0
    num_colors = len(Ramsey)
    choose_table = np.array([[choose(k,n) for n in range(num_vertices+1)] for k in range(Ramsey.max()+1)])
    Colors = np.arange(num_colors)
    Vertices = np.arange(num_vertices)
    Edges = list(it.combinations(Vertices,2))
    #reverse lookup for edges below.  Eg if slot 3 above contains edge (2,5), the the dict below has entry (2,5):3
    Edges_idx = dict((edge, idx) for idx,edge in enumerate(Edges)) 
    num_edges = choose_table[2,num_vertices]

    threads_per_block = 1
    edges_per_clique = np.array([choose_table[2,ram] for ram in Ramsey])
    cliques_per_color = np.array([choose_table[ram,num_vertices] for ram in Ramsey])
    num_cliques_cum = cliques_per_color.cumsum()
    blocks_per_color = np.ceil(cliques_per_color / threads_per_block).astype('uint32')   
    num_blocks = blocks_per_color.sum()
    cliques_per_block = np.ceil(cliques_per_color / blocks_per_color).astype('uint32')
    #The objects below tells each block which color and how many cliques/edges it will monitor.
    #Note each vector is repetitive.  If color 0 gets 7 blocks, the first 7 entries will be the same
    block_color = np.repeat(Colors,blocks_per_color).astype('uint32')
    block_num_cliques = np.repeat(cliques_per_block,blocks_per_color).astype('uint32')
    block_edges_per_clique = np.repeat(edges_per_clique,blocks_per_color).astype('uint32')
    
    edge_idx_to_pair = np.array(list(it.combinations(range(num_vertices),2)))
    edge_pair_to_idx = np.zeros([num_vertices,num_vertices])
    for (idx,e) in enumerate(edge_idx_to_pair):
        edge_pair_to_idx[tuple(e)] = idx
    edge_pair_to_idx += edge_pair_to_idx.T
    np.fill_diagonal(edge_pair_to_idx,num_edges) 
     
    assign_Blocks_to_Cliques = np.full([num_blocks,cliques_per_block.max(),edges_per_clique.max()],
                                       fill_value=num_edges, dtype='uint32')

    #Counters that that tracks the next open block and thread on each block
    next_open_block = 0    
    next_open_thread = np.zeros(num_blocks,dtype='int')
    for color, clique_size in enumerate(Ramsey):
        #Creates a generator to produce all cliques (the list of vertices).
        Cliques = it.combinations(Vertices,clique_size)
        
        #Makes the vector [0,1,2,...,num_blocks-1,0,1,2,...,num_blocks-1,....] of length num_cliques
        assign_Cliques_to_Blocks = np.arange(cliques_per_color[color]) % blocks_per_color[color]
        
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
    
    kernel_code ="""
    #include <stdio.h>
    __global__ void find_problems(int *block_color, int *edges_per_clique, int *edges, unsigned char *coloring, int *Problems, int edges_per_thread, int cliques)
    {
        //__shared__ int shared_coloring[shared_size];

        int clique_idx = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.y*gridDim.x; 
        int color = block_color[clique_idx];
        /*if(threadIdx.x < shared_size)
        {
            shared_coloring[threadIdx.x] = coloring[threadIdx.x];
        }*/
        if(clique_idx < cliques)
        {
            //This weird way of starting and stopping allows us to traverse the 3D-array of assign_Blocks_to_Cliques easily
            int start = clique_idx*edges_per_thread;
            int end = start + edges_per_clique[blockIdx.x];
            int e = start;
            while((e < end) && (coloring[edges[e]] == color))
            {
                e++;
            }
            //Problems[clique_idx] = (e >= end) ? 1 : 0;
            if(e==end)
	    {
		atomicAdd(&Problems[color],1);
	    }	
	}
    }
    """
    
    regex_fixing_dictionary = {"shared_size":(num_edges+1)}

    for key, val in regex_fixing_dictionary.items():
        kernel_code = kernel_code.replace(str(key),str(val))
    mod = SourceModule(kernel_code)
    _, _, edges_per_thread = assign_Blocks_to_Cliques.shape
    edges_per_thread = np.uint32(edges_per_thread)
    block_dimensions = (1,1,1)
    grid_dimensions =(int(num_cliques_cum[-1]**.3333333333)+1, int(num_cliques_cum[-1]**.3333333333)+1, int(num_cliques_cum[-1]**.3333333333)+1)
    func = mod.get_function("find_problems")
    
    #Code below sets up the GPU checker
    block_color, block_color_gpu = mtogpu(block_color, 'int32')
    block_edges_per_clique,block_edges_per_clique_gpu = mtogpu(block_edges_per_clique, 'int32')
    assign_Blocks_to_Cliques,assign_Blocks_to_Cliques_gpu = mtogpu(assign_Blocks_to_Cliques, 'int32') 
    def find_problems_cuda(coloring_gpu):
        Problems_cpu = np.zeros(num_colors)
        Problems_cpu, Problems_gpu = mtogpu(Problems_cpu, 'int32')    
        func(block_color_gpu, block_edges_per_clique_gpu, assign_Blocks_to_Cliques_gpu, coloring_gpu, Problems_gpu, edges_per_thread, np.uint32(cliques_per_color.sum()), block=block_dimensions, grid=grid_dimensions, shared=0)
        print("Problems old: ", Problems_gpu)      
        Problems_cpu = Problems_gpu.get().astype('int32')
        return gpuarray.sum(Problems_gpu).get().astype('int32'), Problems_cpu
    
    kernel_code ="""
    extern "C"
        #include <stdio.h>
    
        __device__ void edge_idx_to_pair_fcn(ushort *edge_idx_to_pair, ushort idx, ushort *v, ushort *w)
        {
            ushort start = 2*idx;
            *v = edge_idx_to_pair[start];
            *w = edge_idx_to_pair[start+1];    
        }

        __device__ void problem_checker(ushort *clr, ushort *n_edges, ushort *edges, unsigned char *coloring, uint *Problems)
        {
            uint block_idx = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
            ushort e = 0; 
            while((e <*n_edges) && (coloring[edges[e]] == *clr))
            {
                e++;
            }
            if (block_idx == PRINT_ID)
            {
                for(int i = 0; i < *n_edges; i++)
                {
  //                  printf("For clique %u with color %u edge %u is %u\\n", block_idx, *clr, i, coloring[edges[i]]);
                }
            }
            
            if(e== *n_edges)
            {
                atomicAdd(&Problems[*clr],1);//atomicAdd(&Problems[block_idx],1);
            }
        }

        __device__ void get_assignment(uint *choose_table, ushort *ramsey, ulong *num_cliques_cum, ushort *edge_idx_to_pair, ushort *edge_pair_to_idx, ushort *clr, uint *clique_idx, ushort *n_verts, ushort *verts, ushort *n_edges, ushort *edges)
        {
            ushort i, j, k, row, col, start;
            uint block_idx = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
            if(block_idx < NUM_CLIQUES)
            {
                //First, determine which color we are.
                (*clr) = 0;
                while(num_cliques_cum[(*clr)] <= block_idx){(*clr)++;}
                (*n_verts) = ramsey[(*clr)];
                (*n_edges) = choose_table[2*(NUM_VERTS+1)+(*n_verts)];
                
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
                    start = row*(NUM_VERTS+1);
                    col = 0;
                    while(choose_table[start+col] <= N){
                        col++;
                    }
                    col--;
                    verts[row-1] = col;
                    N -= choose_table[start+col];
                }
               // printf("block_idx = %5u, color = %1u, n_verts = %2u, n_edges = %3u, clique_idx = %5u\\n", block_idx, (*clr), (*n_verts), (*n_edges), (*clique_idx));
      
    
                //Now,  determine our list of edges.
                k = 0;
                ushort v, w;
                for(i = 0; i < (*n_verts); i++)
                {
                    for(j = i+1; j < (*n_verts); j++)
                    {
                        v = verts[i];
                        w = verts[j];
                        edges[k] = edge_pair_to_idx[v*NUM_VERTS+w];
                        if(block_idx == PRINT_ID)
                        {
                            ushort r,s;
                            edge_idx_to_pair_fcn(edge_idx_to_pair, edges[k], &r, &s);
//                            printf("edge (%u,%u) -> edge_idx %u -> (%u,%u)\\n",v,w,edges[k],r,s);                    
                        }
                        k++;
                    }
                }
            }
        }
    
        __global__ void run(uint *choose_table, ushort *ramsey, ulong *num_cliques_cum, ushort *edge_idx_to_pair, ushort *edge_pair_to_idx, unsigned char *coloring, uint *Problems)
        {
            uint i, start;
            int block_idx = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;        
        
            __shared__ ushort verts[MAX_RAMSEY];
            for(i = 0; i < MAX_RAMSEY; i++){verts[i] = NUM_VERTS;}
        
            __shared__ ushort edges[MAX_EDGES];
            for(i = 0; i < MAX_EDGES; i++){edges[i] = choose_table[2*(NUM_VERTS+1)+NUM_VERTS];} 
            
            ushort clr, n_verts, n_edges;
            uint clique_idx;
            get_assignment(choose_table, ramsey, num_cliques_cum, edge_idx_to_pair, edge_pair_to_idx, &clr, &clique_idx, &n_verts, verts, &n_edges, edges);        
        
            problem_checker(&clr, &n_edges, edges, coloring, Problems);
        }
    """
    regex_fixing_dictionary = {"NUM_VERTS":num_vertices, "MAX_RAMSEY":Ramsey.max(), "MAX_EDGES":edges_per_clique.max(),
                          "NUM_CLIQUES":num_cliques_cum[-1], "PRINT_ID":3}#((2**32)-1)}
    for key, val in regex_fixing_dictionary.items():
        kernel_code = kernel_code.replace(str(key),str(val))
    mod = SourceModule(kernel_code)
    run = mod.get_function("run")
    
    print(num_cliques_cum)
    choose_table, choose_table_gpu = mtogpu(choose_table, 'uint32')
    Ramsey,Ramsey_gpu = mtogpu(Ramsey, 'uint16')
    num_cliques_cum, num_cliques_cum_gpu = mtogpu(num_cliques_cum, 'uint64')
    edge_idx_to_pair,edge_idx_to_pair_gpu = mtogpu(edge_idx_to_pair, 'uint16')
    edge_pair_to_idx, edge_pair_to_idx_gpu = mtogpu(edge_pair_to_idx, 'uint16')
    def find_problems_gen(coloring_gpu):
        Problems_cpu = np.zeros(num_colors)
        Problems_gpu, Problems_gpu = mtogpu(Problems_cpu, 'int32')
        run(choose_table_gpu, Ramsey_gpu, num_cliques_cum_gpu, edge_idx_to_pair_gpu, edge_pair_to_idx_gpu, coloring_gpu, Problems_gpu, block=block_dimensions, grid=grid_dimensions, shared=0)
        print("Problems new", Problems_gpu)
        Problems_cpu = Problems_gpu.get().astype('int32')
        return gpuarray.sum(Problems_gpu).get().astype('int32'), Problems_cpu
    
    def print_problems(Problems):        
        Z = pd.DataFrame(Problems.astype('uint32'))
        Z.insert(0,'problems',Z.sum(axis=1))
        Z.insert(0,'color',block_color)
        Z = Z.T
        problem_idx = np.any(Z,axis=-1)
        problem_idx[:2] = True
        display(Z.ix[problem_idx,:])

    def print_status():
        now = datetime.datetime.now()
        elapsed = now - start_time
        print("%d steps done in %s.  Best coloring so far was step %d with %d problems.  Time now %s."
                   %(step,str(elapsed).split('.')[0],step_best,num_problems_best,str(now).split('.')[0]))
    def increase_red_edges():
        while(list(coloring_cpu).count(red)<min_red):
            idx = [i for i in range(len(coloring_cpu)-1) if coloring_cpu[i]!=red]
            coloring_cpu[random.choice(idx)] = red
    #Initialize the Markov chain
    coloring_cpu = np.random.choice(Colors, size=num_edges+1, replace=True).astype('uint32')
    coloring_cpu[num_edges] = num_colors
    coloring_best = coloring_cpu.copy()
    coloring_cpu, coloring_gpu = mtogpu(coloring_cpu, 'uint8')
    print("coloring is: ",coloring_cpu)
    num_problems_proposed_gen, Problems_proposed_gen = find_problems_gen(coloring_gpu)
    num_problems_proposed_cuda, Problems_proposed_cuda = find_problems_cuda(coloring_gpu) 
    
    if num_problems_proposed_gen == num_problems_proposed_cuda:
        print("They agree!!!!!!!")
    else:
        raise Exception("They disagree")
            
    num_problems_current = num_problems_proposed_gen.copy()    
    num_problems_best = num_problems_proposed_gen.copy()
    
    step = 0
    step_best = step
    
    loop_length = 100000
    loop_step = 0
    loops_done = 0
    start_compute = datetime.datetime.now()
    for i in range(num_steps):
        #if num_problems_best == 0:
        #    break
        
        edge_idx = np.random.randint(0,num_edges)
        color_delta = np.random.randint(1,num_colors)
        edge_color_old = coloring_cpu[edge_idx]
        #edge_color_new = (edge_color_old + color_Deltas[i]) % num_colors
        edge_color_new = (edge_color_old + color_delta) % num_colors
        coloring_cpu[edge_idx] = edge_color_new
        if(list(coloring_cpu).count(red) < min_red):
            increase_red_edges()    
        coloring_cpu, coloring_gpu = mtogpu(coloring_cpu, 'uint8')
        
        num_problems_proposed_gen, Problems_proposed_gen = find_problems_gen(coloring_gpu)
        num_problems_proposed_cuda, Problems_proposed_cuda = find_problems_cuda(coloring_gpu) 
        
        if num_problems_proposed_gen == num_problems_proposed_cuda:
            print("They agree!!!!!!!")
        else:
            raise Exception("They disagree")
        num_problems_diff = num_problems_current - num_problems_proposed_gen
        
        if num_problems_diff >= 0:
             #print("Proposed is better.  Accepting.")            
            num_problems_current = num_problems_proposed_gen.copy()
            #Problems_current = Problems_proposed.copy()
            if num_problems_proposed_gen < num_problems_best:
                step_best = step
                coloring_best = coloring_cpu.copy()
                num_problems_best = num_problems_proposed_gen.copy()
                #Problems_best = Problems_proposed.copy()
                print_status()
        else:            
            accept = np.exp(beta * num_problems_diff)            
            r = np.random.random()
            #print("Proposed is worse.  But I will accept it anyway if I draw a number less than %.3f.  I drew %.3f." % (accept,r))            
            if r <= accept:            
                #print("So I accept the move even though it is worse.")                
                num_problems_current = num_problems_proposed_gen.copy()
                #Problems_current = Problems_proposed.copy()
            else:                
                #print("So I reject.")
                coloring_cpu[edge_idx] = edge_color_old
        step += 1
        loop_step += 1
        if(loop_step >= loop_length):
            loops_done += 1
            loop_step = 0
            print_status()
            compute_time = (datetime.datetime.now() - start_compute).seconds + (datetime.datetime.now() - start_compute).days*60*60*24
            steps_done = loops_done*loop_length
            rate = steps_done / compute_time
            job_time = (num_steps-steps_done)/rate
            m, s = divmod(job_time,60)
            h, m = divmod(m,60)
            d, h = divmod(h,24)
            y, d = divmod(d,365)
            print("At %.0f colorings/second, it'll take me %d years %d days %d hours %d minutes and %d seconds to complete the remaining steps."%
                  (rate,y,d,h,m,s))

    print("FINISHED!!")
    coloring_cpu = coloring_best.copy()
    coloring_cpu, coloring_gpu = mtogpu(coloring_cpu, 'uint8')
    num_problems_best, _   = find_problems_gen(coloring_gpu)
    
    print()
    print_status()
    final_coloring = pd.DataFrame()
    final_coloring['edge'] = Edges
    final_coloring['color'] = coloring_best[:num_edges]
    #display(final_coloring)
    return final_coloring
Ramsey = [3,4]
num_vertices = 5 
num_steps =100 
bill = main_random(Ramsey, num_vertices, num_steps)
#for i in range(9,16):
#    num_vertices = i
#    bill = main_random(Ramsey,num_vertices, num_steps)
#    print("num_vertices is: ", num_vertices)
