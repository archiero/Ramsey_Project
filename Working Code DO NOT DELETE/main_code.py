#%%time
from setup import *
import time
start = time.time()
num_verts = 40
ramsey = [3,10]
beta = 1.2

num_colorings =5 
num_steps = 100
report_period = 20

get_cliques = False
printout = False
print_coloring = 0
np.random.seed(42)

exec(open("count_problems_parallel.py").read())

problem_counts_current = count_problems(colorings,printout)

problem_counts_proposed = problem_counts_current.copy()
problem_counts_best = problem_counts_current.copy()
colorings_best = colorings.copy()
best_step = np.zeros(num_colorings).astype('uint32')


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

print("step:%u,  total time = %f"%(0,(time.time()-start)))
print_status(best_step,problem_counts_best)

print("Begin Markov Chain")
edge_color_old = np.zeros(num_colorings).astype('uint8')
for step in range(1,num_steps+1):
    found = (problem_counts_current == 0)
    if np.any(found):
        found = found.nonzero()
        print("FOUND AT LEAST ONE WITH NO PROBLEMS: %s"%found)
        print("step:%u,  total time = %f"%((step-1),(time.time()-start)))
        print_status(best_step,problem_counts_best)
        print_colorings(found)
        break
        
    change_edge = np.random.randint(0, tot_edges, size=num_colorings).astype('uint16')
    delta_color = np.random.randint(1, num_colors, size=num_colorings).astype('uint8')
    for c in range(num_colorings):
        e = change_edge[c]
        edge_color_old[c] = colorings[c,e]
        colorings[c,e] = ((colorings[c,e] + delta_color[c]) % num_colors)
        
    
    problem_counts_gpu *= 0
    problem_counts_proposed = count_problems(colorings,printout)

    for c in range(num_colorings):
        if(problem_counts_current[c] >= problem_counts_proposed[c]):
            problem_counts_current[c] = problem_counts_proposed[c]
            
            if(problem_counts_best[c] > problem_counts_proposed[c]):
                problem_counts_best[c] = problem_counts_proposed[c]
                colorings_best[c,:] = colorings[c,:].copy()
                best_step[c] = step

        else:
            prob_diff = problem_counts_proposed[c] - problem_counts_current[c]
            accept = np.exp(-1 * beta * prob_diff)
            r = np.random.random()
            if r <= accept:
                #print(" but I accept it anyway.")
                problem_counts_current[c] = problem_counts_proposed[c]
            else:
                #print(" and I reject it.")
                colorings[c,change_edge[c]] = edge_color_old[c]
    if(step % report_period == 0):
        print("step:%u,  total time = %f"%(step,(time.time()-start)))
        print_status(best_step,problem_counts_best)

