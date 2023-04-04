import numpy as np    

all_y_valls = np.array([10, 0, 180, 0,0,0,0,0,0,0,0,0,0,0,0,0])
#all_y_valls = all_y_valls-max(all_y_valls)
print(all_y_valls)
soft_sum = np.sum(np.exp(all_y_valls))  #TODO: overloading is here! check and fix!
print(soft_sum)
y = np.array([np.e**all_y_valls/soft_sum ])
print(y)
print(np.sum(y))