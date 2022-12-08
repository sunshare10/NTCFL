import matplotlib.pyplot as plt
import numpy as np
 
federated=np.load('federated_loss.npy')
local=np.load('local.npy')
fig,ax = plt.subplots()
 
plt.xlabel('communication rounds')
plt.ylabel('Training loss')
 

 

 
 

plt.plot(range(1,101),federated,label="Federated learning")
plt.plot(range(1,101),local,label="Centralized")
 
"""open the grid"""
plt.grid(True)
 
plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
 
plt.show()
