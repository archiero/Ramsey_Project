from setup import *
now = datetime.datetime.now()
open("problem_free.py", mode = "w").write("from setup import *\ncoloring = []")
for i in range(50):
    exec(open("tabu_short_term.py", mode = "r").read())
append_to_file(["coloring = np.asarray(coloring)"], "problem_free.py")
print(datetime.datetime.now() - now)
