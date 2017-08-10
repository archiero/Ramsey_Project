#%%time
#######################################
#This code is the implementation of the short-term tabu search. Simply put, it keeps track of the 100 most recent places that the seach has been and won't let it go back there. It will also check num_colorings neighbors of the current main_coloring and then deal with the best one in an identical fashion to how simulated annealing dealt with the randomly selected neighbor. The combination of these two adaptions should help prevent the code from being stuck in local optimums. Later, we're going to include target-analysis and the longer-term tabu parameters
#######################################
from setup import *
import time
start = time.time()
num_verts = 6 
ramsey = [3,3]
beta = 1.2
THE_ONE_TO_KEEP = np.ones(np.int(num_verts*(num_verts-1)/2))

num_colorings =5 
num_steps = 100
report_period = 20

get_cliques = False
printout = True
print_coloring = 0
np.random.seed(42)

exec(open("count_problems_tabu.py").read())
main_coloring = np.random.randint(0, num_colors, size = tot_edges)
#Recall that the blocks are set up so that each one block checks one clique each and that the different threads check different colorings. So, in order to just check one, we're going to change the block dim to reflect this and then change it back.
#This is the array housing all of the neighbors
colorings = np.zeros((num_colorings, tot_edges)).astype("uint8")

block_dims = (1,1,1)
main_coloring_gpu, main_coloring = mtogpu(main_coloring, 'uint8')

problems_proposed = count_problems(main_coloring_gpu, printout)
block_dims = (num_colorings,1,1)

lowest_proposed = np.max(problems_proposed)
coloring_proposed = np.argmax(problems_proposed)
problems_current = lowest_proposed.copy()
problems_best = lowest_proposed.copy()
colorings_best = main_coloring.copy 
index_best = coloring_proposed.copy()
best_step = 0

def print_colorings(idx=None):
    df = pd.DataFrame(colorings)
    df.columns = [tuple(e) for e in edge_idx_to_pair]
    if idx is not None:
        df = df.loc[idx]    
    display(df)


def print_status(best_step,problem_counts_best):
    df = pd.DataFrame()
    df['best step'] = best_step
    df['problems'] = problem_counts_best
    df = df.T
    idx = np.argmin(problem_counts_best)#.argmin()
    s = 'best='+str(idx)
    df[s] = df[idx]
    display(df)

#print("step:%u,  total time = %f"%(0,(time.time()-start)))
#print_status(best_step,problem_counts_best)
edges_array = np.arange(tot_edges)
tabu_list = np.zeros((2,tot_edges))
tabu_list[0] = main_coloring.copy()
#tabu_list = np.zeros((100,tot_edges))
print("Begin Markov Chain")
edge_color_old = np.zeros(num_colorings).astype('uint8')
for step in range(1,num_steps+1):
    #found = (problem_counts_current == 0)
    print(problems_current)
    if problems_current ==0:#np.any(found):
        #found = found.nonzero()
        print("FOUND AT LEAST ONE WITH NO PROBLEMS: %s"%index_best)
        print("step:%u,  total time = %f"%((step-1),(time.time()-start)))
        print_status(best_step, problems_best)
        print_colorings(index_best)
        THE_ONE_TO_KEEP = coloring_best
        break
        
#    change_edges = np.random.choice(edges_array, size = num_colorings)
#    delta_color = np.random.randint(1, num_colors, size=num_colorings).astype('uint8')    
#    for c in range(num_colorings):
#        e = change_edges[c]
#        edge_color_old[c] = colorings[c,e]
#        colorings[c,e] = ((colorings[c,e] + delta_color[c]) % num_colors)
    change_edges = np.zeros(num_colorings).astype('uint16')
    delta_color = np.zeros(num_colorings).astype('uint8')    
    c = 0
    attempt = 0
    
    while c < num_colorings:
        attempt += 1
        #first, find a random neighbor of the main coloring and save it to the right spot of the colorings array
        change_edges[c] = np.random.choice(edges_array, size = 1).astype('uint16')
        delta_color[c] = np.random.randint(1,num_colors, size = 1).astype('uint8')
        e = change_edges[c]
        colorings[c] = main_coloring
        edge_color_old[c] = colorings[c,e]
        colorings[c,e] = ((colorings[c,e] + delta_color[c]) % num_colors)
	#First, check to see if the neighbor is on the tabu list
        if(np.any([np.array_equal(colorings[c],tabu_list[i]) for i in range(len(tabu_list))]) == False):
            #Now, check to see if this neighbor has been grabbed before:
            if(np.any([np.array_equal(colorings[c], colorings[i]) for i in range(c)]) == False):
                c+= 1
        #If either of these conditions fail, find another random neighbor
        if attempt >= 10:
            raise Exception("SOMETHINGS UP WITH THE TABU LIST")
    #Note: at any given time, for R(3,10), tabu_list of size 100 and 32 neighbors to check, there is at most a (100 + 31)/780 = 16.8% chance of the process repeating itself. 
    problem_counts_gpu *= 0
    problems_proposed = count_problems(colorings,printout)
    lowest_proposed = np.min(problems_proposed)
    coloring_proposed = np.argmin(problems_proposed)
   
    if(lowest_proposed <= problems_current):
         main_coloring = colorings[coloring_proposed].copy()
         problems_current = lowest_proposed.copy()
         if problems_best >= lowest_proposed:
             problems_best = lowest_proposed.copy()
             best_coloring = colorings[coloring_proposed].copy()
             best_step = step.copy()
             index_best = coloring_proposed.copy()
    else:
         prob_diff = problems_current = lowest_proposed
         accept = np.exp(-1 * beta * prob_diff)
         rand = np.random.random()
         if rand <= accept:
             main_coloring = colorings[coloring_proposed].copy()
             problems_current = lowest_proposed.copy()
    #Recall, we start with tabu_list[0] = first coloring and step = 1. So, to have tabu_list[1] be equal to coloring after we've taken the first step and then update at the appropiate point, have it to be saved here. 
    tabu_list[step%len(tabu_list)] = main_coloring.copy()
    
    if(step % report_period == 0):
        print("step:%u,  total time = %f"%(step,(time.time()-start)))
        print_status(best_step,problem_counts_best)

