from setup import *
from scipy.special import binom as choose

num_vertices = 40
ramsey = [3,10]
num_steps = 1000
beta = .05
training_rows = 10#int(choose(num_vertices,2)/4)
start = time.time()
get_cliques = False
report_period = 100
#This has the method for counting problems saved. Also a couple of useful functions like a print_status function
#exec(open("count_problems_pandas.py", mode = 'r').read())
exec(open('count_problems_tabu.py', mode = 'r').read())
tabu_length = training_rows

num_colors = len(ramsey)
num_edges = int(choose(num_vertices,2))
Colors = list(range(num_colors))

main_coloring = np.random.choice(Colors, size=num_edges+1, replace=True).astype('uint8')
main_coloring[num_edges] = num_colors
edges_array = np.arange(num_edges)

#First, we're going to create a way to make all possible neighbors
color_changes = np.array(range(1,num_colors))
edges = np.array(range(num_edges))
possible_changes = list(it.product(color_changes, edges_array))
#This starts the search
problems_proposed = count_problems(main_coloring)
problems_current = np.min(problems_proposed)
problems_best = np.min(problems_proposed)
coloring_best = main_coloring.copy() 
best_step = 0
step = 0

def print_status():
    now = time.time()
    elapsed = now - start
    print("%d steps done in %s.  Best coloring so far was step %d with %d problems.  Time now %s."
           %(step,str(elapsed).split('.')[0],best_step,problems_best,str(now).split('.')[0]))

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

#This handles all of the short, medium and long term aspirations
#    def aspiration_adjustments(colorings, problems):
#        for c in range(len(colorings)):
#            coloring = colorings[c]
#            tabu_indicator = [np.array_equal(coloring,tabu_list[i]) for i in range(tabu_length)]
#            if np.any(tabu_indicator) == True:
#                problems[c] *=100 
#        return(problems)

print("Begin Gibb's Sampler")

for step in range(1,num_steps+1):
    print(step)
    #print(problems_current)
    if problems_current == 0:
        print("step:%u,  total time = %f"%((step-1),(time.time()-start)))
        print_status()
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
    #drop the neighbors that have already been examined from the tabu list
    #Note: don't edit the except statement. If a tabu element isn't in neighbors it will always raise a 
    #ValueError so we expect that. Otherwise, we want to know about errors.
    neighbors = list(neighbors)
    for tabu in tabu_list:
        try:
            neighbors.remove(tabu)
        except ValueError:
            pass
    neighbors = np.array(neighbors)
    #Now, predict all of the problems and use that to keep some to examine exactly
    #neighbor_problems = model.predict(neighbors)
    #order = np.argsort(neighbor_problems)
    order = np.random.choice(range(len(possible_changes)), size = training_rows)
    examined_neighbors = neighbors[order[0:training_rows]]
    
    #Find all of the exact problems then retrain the neural net and back it up.
    training_target = count_problems(neighbors[1:training_rows])
    training_data = examined_neighbors.copy()
    #for i in range(training_rows):
    #    training_data[i] = examined_neighbors[i]
    #    training_target[i] = count_problems(examined_neighbors[i])
    
    for i in range(training_rows):
        target = ''.join(['target.append(', str(training_target[i]),')'])
        data = ''.join(['data.append(', str(training_data[i]).replace(" ", ','), ')'])
        append_to_file([data, target], 'NN_training.py')
    
    #model.train_on_batch(training_data, training_target)
    actual_problems = training_target.copy()
    #This keeps the order of the truly examined neighbors in terms of number of actual problems
    coloring_order = np.argsort(actual_problems)
    #Make the lowest number of problems the one to keep
    problems_proposed = actual_problems[coloring_order[0]]
    problems_diff = np.int(problems_current) - np.int(problems_proposed)
    print('prob_diff',problems_diff)
    if problems_diff >= 0 :
#             print("problems_current: %u lowest_proposed: %u step %u"%(problems_current, lowest_proposed, step))
        main_coloring = np.ravel(examined_neighbors[coloring_order[0]].copy())
        problems_current = problems_proposed.copy()
        if problems_best > problems_proposed :
#                 print("We have a new best!!!!!!! The previous happened on step %u"%(best_step))
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
