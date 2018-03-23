#%%time
#######################################
#This code is the implementation of the short-term tabu search along with the neural network to improve the search
#######################################
from setup import *
from scipy.special import binom as choose
import time

start = time.time()
num_verts = 22
ramsey = [3,7]
beta = .05
tabu_length = 30
training_rows = np.int(choose(num_verts,2)/5)#64 
num_steps = 1*(10**1) 
report_period = 5

#This has the method for counting problems saved. Also a couple of useful functions like a print_status function
exec(open("count_problems_pandas.py", mode = 'r').read())
#exec(open('count_problems_tabu.py', mode = 'r').read())

get_cliques = False 
printout = True
print_coloring = 0
#np.random.seed(142)
num_colors = len(ramsey)
num_edges = int(choose(num_verts,2))

main_coloring = np.random.choice(Colors, size=num_edges+1, replace=True).astype('uint8')
main_coloring[num_edges] = num_colors
edges_array = np.arange(num_edges)

#First, we're going to create a way to make all possible neighbors
color_changes = np.array(range(1,num_colors))
edges = np.array(range(num_edges))
possible_changes = list(it.product(color_changes, edges_array))
#This starts the Markov chain's search
problems_proposed = count_problems(main_coloring)
problems_current = problems_proposed.copy()
problems_best = problems_proposed.copy()
coloring_best = main_coloring.copy() 
best_step = 0
step = 0

print("step:%u,  total time = %f"%(0,(time.time()-start)))
print_status()
#Creating the tabu list and prepping the neural network
tabu_list = np.zeros((tabu_length,len(main_coloring)))
tabu_list[0] = main_coloring.copy()
training_data = np.zeros((training_rows, len(main_coloring)))
training_target = np.zeros(training_rows)
model = reload_model()
#This is an array for all possible neighbors and their number of problems
neighbors = np.zeros((len(possible_changes), len(main_coloring))).astype("int")
neighbor_problems = np.zeros(len(possible_changes))
examined_neighbors = np.zeros((training_rows, len(main_coloring)))
actual_problems = np.zeros(training_rows)
#This handles all of the short, medium and long term aspirations
def aspiration_adjustments(colorings, problems):
    for c in range(len(colorings)):
        coloring = colorings[c]
        tabu_indicator = [np.array_equal(coloring,tabu_list[i]) for i in range(tabu_length)]
        if np.any(tabu_indicator) == True:
            problems[c] *=100 
    return(problems)

print("Begin Markov Chain")

for step in range(1,num_steps+1):
    print(step)
    #print(problems_current)
    if problems_current == 0:
        print("step:%u,  total time = %f"%((step-1),(time.time()-start)))
        print_status(best_step, problems_best,step)
        print_main_coloring()
        break
    
    #First, create all of the neighbors. 
    for i in range(len(possible_changes)):
        change = possible_changes[i]
        neighbors[i] = main_coloring.copy()
        edge_idx = change[0]
        color_delta = change[1]
        edge_color_old = main_coloring[edge_idx]
        edge_color_new = (edge_color_old + color_delta) % num_colors
        neighbors[i][edge_idx] = edge_color_new
    #Now, predict all of the problems and use that to keep some to examine exactly
    neighbor_problems = model.predict(neighbors)
    order = np.argsort(neighbor_problems)
    examined_neighbors = neighbors[order[0:training_rows]]
    #Find all of the exact problems then retrain the neural net and back it up.
    #training_target = count_problems(neighbors[1:training_rows
    #training_data = neighbors[1:training_rows]
    for i in range(training_rows):
        training_data[i] = examined_neighbors[i]
        training_target[i] = count_problems(examined_neighbors[i])
    model.train_on_batch(training_data, training_target)
    backup_model(model)    
    actual_problems = training_target.copy()
    actual_problems = aspiration_adjustments(examined_neighbors, actual_problems)
    #This keeps the order of the truly examined neighbors in terms of number of actual problems
    coloring_order = np.argsort(actual_problems)
    #Make the lowest number of problems the one to keep
    problems_proposed = actual_problems[coloring_order[0]]
    problems_diff = problems_current - problems_proposed
    print(problems_diff)
    if problems_diff >= 0 :
#         print("problems_current: %u lowest_proposed: %u step %u"%(problems_current, lowest_proposed, step))
         main_coloring = np.ravel(examined_neighbors[coloring_order[0]].copy())
         problems_current = problems_proposed.copy()
         if problems_best > problems_proposed :
#             print("We have a new best!!!!!!! The previous happened on step %u"%(best_step))
             problems_best = problems_proposed.copy()
             coloring_best = examined_neighbors[coloring_order[0]].copy()
             best_step = step.copy()          
    else:
        accept = np.exp(problems_diff*beta)
        Maryam = np.random.random()
        #print("It wasn't better. If I get below %u, I accept and I got %u"%(accept, Maryam))
        if Maryam <= accept:
            print("I accepted")
            main_coloring = examined_neighbors[coloring_order[0]].copy()
            problems_current = problems_proposed
    #Recall, we start with tabu_list[0] = first coloring and step = 1. So, to have tabu_list[1] be equal to coloring after we've taken the first step and then update at the appropiate point, have it to be saved here. 
    tabu_list[step%len(tabu_list)] = main_coloring.copy()
    if(step % report_period == 0):
        print("step:%u,  total time = %f"%(step,(time.time()-start)))
        print_status()
