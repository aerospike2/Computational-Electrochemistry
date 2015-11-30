# --------------------------------------------------------------------------------------------- #
# ----------------------- CYCLIC VOLTAMMETRY FOR REVERSIBLE REACTION -------------------------- # 
# --------------------------------------------------------------------------------------------- #

# Abhishek Sharma, BioMIP, Ruhr University, Bochum
# 16th November, 2015

"""                                
   The electrochemical reaction was given by,
   
                                     O + e- <--> R
    
   In the python code, we simulate a cyclic voltammogram of a reversible reaction, involving two species (O and R). As the reaction
   is reversible, concentration of the species in the vicinity of the electrode surface is given by Nernst equation. Hence, we can 
   write the concentration of the species as,
   
   Co[0,t]/Cr[0,t] = exp((nF/RT)*(E-E0)) = zeta

   Co[0,t] = zeta/(1+zeta)
   
   Cr[0,t] = 1/(1+zeta)
   
   zeta[k] =  exp((nF/RT)*(E-E0) - tau*(k-1))
   
"""

import time
import numpy as np
import math as mt
import scipy
import scipy.sparse
from scipy.sparse import linalg
from scipy.linalg import solve
import matplotlib
matplotlib.use("GTKAgg") 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ----------------------------------------------------------------------------- #
# Physical Constants

F = 96485.33289		# Faraday's constant (in SI units)
R = 8.3144598           # Molar Gas constant (in SI units)
T = 298.		# Working temperature (in Kelvin)
Dc = 1.e-5		# Diffusion constant (in SI units)
alpha = 0.5		# Transfer coefficient

# Simulation variables

lowerLimit = -8. 	# limit = fx(E-E0)
upperLimit = 8.		# limit = fx(E-E0)
t = 2.*(upperLimit + abs(lowerLimit))
n = 401
D = 2.
tau = t/(n-1)
m = 1 + mt.ceil(6*mt.sqrt(D*(n-1)))

# ----------------------------------------------------------------------------- #
# Creating tridigonal matrix

mat1 = np.zeros(m-2)
mat2 = np.zeros(m-3)
mat3 = np.zeros(m-3)
mat4 = np.zeros(m-2)

mat1[:], mat2[:], mat3[:], mat4[:] = -2., 1., 1., 1.

A = scipy.sparse.diags(diagonals=[mat1, mat2, mat3],
                       offsets=[0, -1, 1], shape=(m-2, m-2),
                       format='csr')  

dMat = scipy.sparse.diags(diagonals=[mat4],
                       offsets=[0], shape=(m-2, m-2),
                       format='csr') - D*A 
                                                                                                           
# print dMat.todense()

# ----------------------------------------------------------------------------- #
# Solving iteratively and plotting data

initialC = np.ones(m)
conc = initialC
dataConc = conc

for k in range(int(n)):
    print 'time step : ', k
    if k > (n+1)/2.:
       zeta = mt.exp(upperLimit - t + tau*(k-1))
    else:
       zeta = mt.exp(upperLimit - tau*(k-1))

    b = conc[1:m-1]
    b[0] += D*zeta/(1+zeta)
    b[m-3] += D
    
    conc[0] = zeta/(zeta+1)
    conc[1:m-1] = scipy.sparse.linalg.spsolve(dMat, b)
    conc[m-1] = 1.
    dataConc = np.append(dataConc, conc, axis=0)
    
dataConc = np.reshape(dataConc, (402, 171)) 

# ----------------------------------------------------------------------------- #
# Plotting concentration profile and calculating current   

# plotting time dependent current 
current = np.empty(len(dataConc))
for i in range(len(dataConc)):
    current[i] = (3.*dataConc[i,0] - 4.*dataConc[i,1] + dataConc[i,2])*(mt.sqrt(D*(n-1)))/(mt.sqrt(4*t))

plt.plot(current)
plt.xlabel('voltage')
plt.ylabel('current')
plt.show()    

# plotting CV
xD = np.empty(len(dataConc))
yD = current
 
for j in range(len(current)):
    if j > (n+1)/2:
       xD[j] = upperLimit - t + (j-1)*tau
    else:
       xD[j] = upperLimit - (j-1)*tau   

plt.xlabel('voltage')
plt.ylabel('current')
plt.plot(xD, yD, 'r')
plt.show()
       
# -------------------------------------------- end - of - file ----------------------------------- #
