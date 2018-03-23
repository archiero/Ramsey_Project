#This code runs the tabu_short_terms, appends them to the the cliques_to_data_mine.py then saves it as an numpy array where every entry is a problem-free coloring as an numpy array. Effectively, this writes stuff I want to learn about to the file I want to do stuff in. 

from setup import *
now = datetime.datetime.now()
open("cliques_to_data_mine.py", mode = "w").write("from setup import *\ncoloring = []")
for i in range(150):
    exec(open("tabu_short_term.py", mode = "r").read())
append_to_file(["coloring = np.asarray(coloring)"], "cliques_to_data_mine.py")
print(datetime.datetime.now() - now)
