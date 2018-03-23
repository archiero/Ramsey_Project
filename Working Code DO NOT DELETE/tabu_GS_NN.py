#%%time
#######################################
#This code is the implementation of the short-term tabu search along with the neural network to improve the search
#######################################
from setup import *
from scipy.special import binom as choose
import time

start = time.time()
num_verts = 5
ramsey = [3,3]
tabu_length = 30
training_rows = np.int(choose(num_verts,2)/5)#64
num_steps = 2*(10**0) 
report_period = 25

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
#This starts the search
problems_current = count_problems(main_coloring)
problems_best = problems_current.copy()
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
#model = reload_model()
#This is an array for all possible neighbors and their number of problems
neighbors = np.zeros((len(possible_changes), len(main_coloring))).astype("int")
neighbor_problems = np.zeros(len(possible_changes))
examined_neighbors = np.zeros((training_rows, len(main_coloring)))
actual_problems = np.zeros(training_rows)
#This handles all of the medium and long term aspirations
#def aspiration_adjustments(colorings, problems):
#    for c in range(len(colorings)):
#        coloring = colorings[c]
#        tabu_indicator = [np.array_equal(coloring,tabu_list[i]) for i in range(tabu_length)]
#        if np.any(tabu_indicator) == True:
#            problems[c] *=100 
#    return(problems)

print("Begin searching")

for step in range(1,num_steps+1):
    print('step')
    if problems_best == 0:
        print("step:%u,  total time = %f"%((step-1),(time.time()-start)))
        print_status()
        print(coloring_best)
        #break
    
    #First, create all of the neighbors. 
    for i in range(len(possible_changes)):
        change = possible_changes[i]
        neighbors[i] = list(main_coloring.copy())
        edge_idx = change[1]
        color_delta = change[0]
        edge_color_old = main_coloring[edge_idx]
        edge_color_new = (edge_color_old + color_delta) % num_colors
        neighbors[i][edge_idx] = edge_color_new
        if neighbors[i] in tabu_list:
            print(neighbors[i])
    #This removes all of the neighbors that are on the tabu list, preventing us from going there
    neighbors= list(neighbors)
    for tabu in tabu_list:
        #print(tabu)
        try:
            neighbors.remove(tabu)
            print("World")
        except ValueError:
            pass
    neighbors = np.asarray(neighbors)
    #print('tabu', tabu_list[0])
    print(tabu_list[0] in neighbors)
    #Now, predict all of the problems and use that prediction to estimate which you examine precisely 
    #neighbor_problems = model.predict(neighbors)
    #order = np.argsort(neighbor_problems)
    examined_neighbors = neighbors[0:training_rows]
    #Find all of the exact problems then retrain the neural net and back it up.
    #training_target = count_problems(neighbors[1:training_rows])
    #training_data = neighbors[1:training_rows]
    for i in range(training_rows):
        #training_data[i] = examined_neighbors[i]
        actual_problems[i] = count_problems(examined_neighbors[i])
    #model.train_on_batch(training_data, training_target)
    #backup_model(model)    
    actual_problems = training_target.copy()
    #Now, decide which one to move to based on the number of problems with the most likely coloring being the one with the 
    #fewest number of problems and every coloring having a non-zero chance. 
    less = [np.max(actual_problems)-problem+1 for problem in actual_problems]
    prob = np.cumsum([problems/(np.sum(less)) for problems in less])
    randy = np.random.rand()
    one_to_move_to = np.digitize(randy, prob)
    main_coloring = np.ravel(examined_neighbors[one_to_move_to].copy())
    problems_current = actual_problems[one_to_move_to]
    if problems_best > problems_current:
#        print("We have a new best!!!!!!! The previous happened on step %u"%(best_step))
        problems_best = problems_current.copy()
        coloring_best = examined_neighbors[one_to_move_to].copy()
        best_step = step
    #Recall, we start with tabu_list[0] = first coloring and step = 1. So, to have tabu_list[1] be equal to coloring after we've taken the first step and then update at the appropiate point, have it to be saved here. 
    tabu_list[step%tabu_length] = main_coloring.copy()
    print(problems_current)
    if(step % report_period == 0):
        print("step:%u,  total time = %f"%(step,(time.time()-start)))
        print_status()
