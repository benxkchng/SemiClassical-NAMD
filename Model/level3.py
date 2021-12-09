import numpy as np
from numpy import array as A

# def model(M=3):

#     #        |  M0 |  M1 |  M2  |  M3 |  M4 |  M5 |
#     ε   = A([  0.0,   0.0,  1.0,  1.0,  0.0,  5.0  ])
#     ξ   = A([ 0.09,  0.09,  0.1,  0.1,  2.0,  4.0  ])
#     β   = A([  0.1,   5.0, 0.25,  5.0,  1.0,  0.1  ])
#     ωc  = A([  2.5,   2.5,  1.0,  2.5,  1.0,  2.0  ])
#     Δ   = A([  1.0,   1.0,  1.0,  1.0,  1.0,  1.0  ])
#     N   = A([  100,   100,  100,  100,  400,  400  ])

#     return ε[M], ξ[M], β[M], ωc[M], Δ[M], N[M]  


# def bathParam(ξ, ωc, ndof):
    

#     ωm = 4.0
#     ω0 = ωc * ( 1-np.exp(-ωm) ) / ndof
#     c = np.zeros(( ndof ))
#     ω = np.zeros(( ndof ))
#     for d in range(ndof):
#         ω[d] =  -ωc * np.log(1 - (d+1)*ω0/ωc)
#         c[d] =  np.sqrt(ξ * ω0) * ω[d]  
#     return c, ω


class parameters():
   NSteps = 4000 #int(2*10**6)
   NTraj = 1
   dtN = 1.0
   dtE = dtN/100
   NStates = 3
   M = 1
   initState = 2
   nskip = 10

   # ndof = 100
   
   au = 27.2114
   ω_c = 0.1/au
   ω_0 = 1.0/au
   gc = 0.05/au
   delE = 1/au
   λ = 0.1/au   # 0.5*ω_0**2*R_0**2
   R0  = (2*λ/ω_0 **2.0)**0.5
   β = 1/(0.0259*au)
   
   # ε, ξ, β, ωc, Δ, ndof = model(3) # model3
   #c, ω  = bathParam(ξ, ωc, ndof)



def Hel(R):
    ω_c = parameters.ω_c
    ω_0 = parameters.ω_0 
    gc = parameters.gc
    delE = parameters.delE
    R0 = parameters.R0


    VMat = np.zeros((3,3))
    
    # Note: In atomic units (a.u), planck constant is unity
    # diagonal terms
    VMat[0,0] = 0
    VMat[1,1] = ω_c
    VMat[2,2] =  0.5*ω_0**2*R0**2  - ω_0**2*R* R0 + delE 

    # off-diagonal terms
    VMat[1,2] , VMat[2,1] = gc, gc
   
   
    return VMat


def dHel0(R):
    # c = parameters.c
    ω_0 = parameters.ω_0
    dH0 = ω_0**2 * R 
    return dH0


def dHel(R):
    R0 = parameters.R0
    ω_0 = parameters.ω_0
    
    dHij = np.zeros((3,3,len(R)))
    # dHij[0,0,:] = c   
    dHij[2,2,:] = - ω_0**2*R0
    return dHij

def initR():
    R0 = 0.0
    P0 = 0.0
    β  = parameters.β
    ω = parameters.ω_0
    # ndof = parameters.ndof

    sigP = np.sqrt( ω / ( 2 * np.tanh( 0.5*β*ω ) ) )
    sigR = sigP/ω
    
    # R = np.zeros(( ndof ))
    # # print(R[0])
    # P = np.zeros(( ndof ))
    # for d in range(ndof):
    #     print('test')
    #     R[d] = np.random.normal()*sigR[d]  
    #     print('R[d] is ',R[d])

    #     P[d] = np.random.normal()*sigP[d]  
    # return R, P

    R = np.random.normal()*sigR + R0
    P = np.random.normal()*sigP + P0
    return np.array([R]), np.array([P])