from setup import *

exec(open("cliques_to_data_mine.py", mode = "r").read())
count_red = [list(coloring[i]).count(0) for i in range(len(coloring))]
count_blue = [list(coloring[i]).count(1) for i in range(len(coloring))]
counts = list(zip(count_red,count_blue))
ratio = [count_blue[i]/count_red[i] for i in range(len(coloring))]

print("Red counts basic stats: Mean %.3f Standard Deviation %.3f Min %.3f Max %.3f"%(np.mean(count_red),np.std(count_red),np.min(count_red),np.max(count_red)))
print("Blue counts basic stats: Mean %.3f Standard Deviation %.3f Min %.3f Max %.3f"%(np.mean(count_blue),np.std(count_blue),np.min(count_blue),np.max(count_blue)))
print("Ratio basic stats: Mean %.3f Standard Deviation %.3f Min %.3f Max %.3f"%(np.mean(ratio), np.std(ratio), np.min(ratio), np.max(ratio)))
