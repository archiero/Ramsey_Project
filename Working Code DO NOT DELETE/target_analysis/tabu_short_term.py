#%time
#######################################
#This code is the implementation of the short-term tabu search. Simply put, it keeps track of the 100 most recent places that the seach has been and won't let it go back there. It will also check num_colorings neighbors of the current main_coloring and then deal with the best one in an identical fashion to how simulated annealing dealt with the randomly selected neighbor. The combination of these two adaptions should help prevent the code from being stuck in local optimums. Later, we're going to include target-analysis and the longer-term tabu parameters
#######################################
from setup import *
import time
start = time.time()
num_verts = 30
ramsey = [3,10]

beta = 1.2
tabu_length = 30
THE_ONE_TO_KEEP = np.ones(np.int(num_verts*(num_verts-1)/2))

num_colorings = 32 
num_steps = 5*(10**5)
report_period = 100

get_cliques = False 
printout = False 
double_check = False
collect_colorings = True
print_coloring = 0
#np.random.seed(142)

exec(open("count_problems_tabu.py").read())

main_coloring = np.random.randint(0, num_colors, size = tot_edges).astype('uint8')
#This is the array housing all of the neighbors
colorings = np.zeros((num_colorings, tot_edges)).astype("uint8")
#We're to make the first coloring in colorings the main_coloring. This way, we can count the problems for it easily with changing the kernel call at all
colorings[0] = main_coloring
colorings[-1] = np.ones(tot_edges).astype('uint8')
problems_proposed = count_problems(colorings, printout)
#This is an array making sure that the problem checking algorithm is working fine. It sends down one blue monochromatic and one red monochromatic and makes sure it comes up with the right number of problems. 
print(problems_proposed[-1], problems_proposed[-2])
print(binom(num_verts,ramsey[-1]),binom(num_verts,ramsey[-2]))
lowest_proposed = problems_proposed[0]
coloring_proposed = 0

problems_current = lowest_proposed.copy()
problems_best = lowest_proposed.copy()
coloring_best = main_coloring.copy() 
best_step = 0
step = 0

def print_main_coloring():
    df = pd.DataFrame(main_coloring)
    df = df.T
    df.columns = [tuple(e) for e in edge_idx_to_pair]
    display(df)

def print_status(best_step,problems_best,step):
    df = pd.DataFrame()
    total_time = time.time() - start
    df['best step'] = [best_step]
    df['problems'] = [problems_best]
    df['rate'] = [total_time/(step+1)]
    display(df)

print("step:%u,  total time = %f"%(0,(time.time()-start)))
print_status(best_step,problems_best,step)

edges_array = np.arange(tot_edges)
tabu_list = np.zeros((tabu_length,tot_edges))
tabu_list[0] = main_coloring.copy()

print("Begin Markov Chain")
edge_color_old = np.zeros(num_colorings).astype('uint8')

for step in range(1,num_steps+1):
    if problems_current == 0:             
        print("step:%u,  total time = %f"%((step-1),(time.time()-start)))
        print_status(best_step, problems_best,step)
        print_main_coloring()
        THE_ONE_TO_KEEP = coloring_best
        #This will record the ramsey number we're looking for and the number of vertices as a comment above the coloring claiming to be problem free. 
        comment = "#Ramsey is RAMSEY and num_verts is NUM_VERTS"
        comment_fixing_dictionary = {"RAMSEY":ramsey, "NUM_VERTS":num_verts}
        for key, value in comment_fixing_dictionary.items():
            comment = comment.replace(str(key),str(value))\
        #This will save the main coloring as an np.array under the name coloring to the appropiate places. It takes main_coloring turns it into a string of the form "[1 0 1 ... 0]", replaces the spaces with commas (thus this will literally write to a .py file [1,0,1,...,0] a.k.a. a list of the appropiate values and length) and attaches a command to save it as an np.array for dumbsey to process. 
        #append_to_file takes all of the objects in the list and appends it to a new line in the appropiate file
        if double_check == True:
            main_coloring = "".join(["coloring = np.array(" , str(main_coloring).replace(" ",",") , ")"])
        elif double_check == False:
            main_coloring = "".join(["coloring.append(np.array(", str(main_coloring).replace(" ",",") , "))"])
        #The reason why the following simpler, neater line was replaced is because, as far as I have been able to tell, numpy has appearance function imbedded. So, [3,3] prints out as a string exactly as "[3 3]" but [3,10] prints as "[ 3 10]" and, in similar fashion, [3,100] prints as "[  3 100]". Not sure why but it does and it's the reason why the for-loop is currently running. 
        #ramsey = "".join(["ramsey = ",str(ramsey).replace(" ",",")])
        Ramsey = ["ramsey = ["]
        for ram in ramsey:
            Ramsey.append(str(ram))
            Ramsey.append(",")
        Ramsey[-1] = "]"
        ramsey = "".join(Ramsey)
        num_verts = "".join(["num_verts = ",str(num_verts)])
        print(ramsey)
        print(num_verts)
        #This command setups the problem_free.py file and saves as the very first line "from setup.py import *"
        if double_check == True:
            open("cliques_to_data_mine.py", mode = "w").write("from setup import *")
        append_to_file([comment,ramsey,num_verts,main_coloring],file="cliques_to_data_mine.py")
        if double_check == True:
           append_to_file([open("double_check_coloring.py", mode = "r").read()], file = "cliques_to_data_mine.py")   
        break
        
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
        if(np.any([np.array_equal(colorings[c],tabu_list[i]) for i in range(tabu_length)]) == False):
            #Now, check to see if this neighbor has been grabbed before:
            if(np.any([np.array_equal(colorings[c], colorings[i]) for i in range(c)]) == False):
                c+= 1
                #If either of these conditions fail, c doesn't go up one and we find another random neighbor
        #Note, I put it here so we wouldn't ever get stuck in an endess while-loop and there's a minimal chance of this condition ever breaking the command because the loop just happened to take a long time.
        if attempt >= tabu_length*num_colorings:
            raise Exception("SOMETHINGS UP WITH THE TABU LIST")
    #Count all of the problems in the randomly generate neighbors and keep the one with the lowest problems 
    problem_counts_gpu *= 0
    problems_proposed = count_problems(colorings,printout)
    lowest_proposed = np.min(problems_proposed)
    coloring_proposed = np.argmin(problems_proposed)
   
    if(lowest_proposed <= problems_current):
#         print("problems_current: %u lowest_proposed: %u step %u"%(problems_current, lowest_proposed, step))
         main_coloring = colorings[coloring_proposed].copy()
         problems_current = lowest_proposed.copy()
         if problems_best > lowest_proposed:
#             print("We have a new best!!!!!!! The previous happened on step %u"%(best_step))
             problems_best = lowest_proposed.copy()
             coloring_best = colorings[coloring_proposed].copy()
             best_step = step           
    else:
         prob_diff = problems_current = lowest_proposed
         accept = np.exp(-1 * beta * prob_diff)
         Maryam = np.random.random()
#         print("It wasn't better. If I get below %u, I accept and I got %u"%(accept, Maryam))
         if Maryam <= accept:
#             print("I accepted")
             main_coloring = colorings[coloring_proposed].copy()
             problems_current = lowest_proposed.copy()
    #Recall, we start with tabu_list[0] = first coloring and step = 1. So, to have tabu_list[1] be equal to coloring after we've taken the first step and then update at the appropiate point, have it to be saved here. 
    tabu_list[step%len(tabu_list)] = main_coloring.copy()
    
    if(step % report_period == 0):
        print("step:%u,  total time = %f"%(step,(time.time()-start)))
        print_status(best_step, problems_best,step)
if(double_check == True):
    exec(open("problem_free.py").read())
