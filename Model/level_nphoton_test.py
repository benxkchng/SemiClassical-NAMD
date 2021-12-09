import numpy as np
from numpy import array as A

# Basis :
# |g g g g g g > | 0 0 0 0 0 0 >  => |G, 0>
# |e g g g g g > | 0 0 0 0 0 0 >  => |E1,0>
# |g e g g g g > | 0 0 0 0 0 0 >  => |E2,0>
# so on
# |g g g g g g > | 1 0 0 0 0 0 >  => |G, F0>
# |g g g g g g > | 0 1 0 0 0 0 >  => |G, F1>
# so on 

def getParameters(N, Jω):
    ω = np.linspace(0.00001,0.1,10000)/27.2114
    dω = ω[1] - ω[0]
    Jω = Jω(ω)
    Fω = np.zeros((len(ω)))
    for iω in range(len(ω)):
        Fω[iω] = (4.0/np.pi) * np.sum( Jω[:iω]/ω[:iω] ) * dω
    λs = Fω[-1] 
    #print (λs * 27.2114)
    ωj = np.zeros((N))
    cj = np.zeros((N))
    for i in range(N):
        j = i+1
        ωj[i] = ω[np.argmin(np.abs(Fω - ((j-0.5)/N) *  λs))] 
        cj[i] = 2.0 * ωj[i] * (λs/(2.0 * N))**0.5 

    # dj = 0.5 * cj**2.0 / ωj**3
    dj = cj / ωj**2
    #dj =np.zeros((N))
    return ωj, dj

# Get LA phonons

#ωLA, dLA = getParameters(40, LA) # 40 accoustic modes

#np.savetxt( "ω.txt", ωLA)
#np.savetxt( "d.txt", dLA)


 
#λs =  (np.sum( 0.5 * (ωLA**2.0) * (dLA**2) * 27.2114) )
#print(f"Total Reorganization Energy = {λs} eV")

def LA(ω):
    ps = 41341.374575751
    a = 1.41586e+12
    α  =  (2.4 * ps**2.0)
    r  =  α /a
    # spectral density
    
    ωb = 0.00223/27.2114
    #* 4.0 * np.pi **2.0 
    return  α * (ω ** 3.0) * np.exp(-ω**2.0/(2.0 * ωb**2.0))

class parameters():
   NSteps = 4000*10 #int(2*10**6)
   NTraj = 1
   dtN = 1.0/10
   dtE = dtN/20
   Nmodes = 5
   Nmols =  2
   NStates = 1 +  Nmols + Nmodes
   M = 1
   initState = 5  # photon mode 2.44 eV
   # 11/18/2021: Add code here that allows initState to be a polariton state with definite wavevector
   nskip = 50

   ωc0   = 2.4/27.2114 # lowest photon freq  Use 3.81 from Feist paper
   c     = 137.0 
   kx    = ωc0/c 
   # Comment: 11/19/2021 - We can rewrite the Lz in terms of maximum kz value that we want; this defines a energy difference
   Lz    = (10*10**-6) * Nmols/(0.528*10**-10) # 5*10**-6/(0.528*10**-10)  #100 * Nmols # Nmolecule in Lz Figure out
   kz    = 2 * np.pi *  np.arange(0,Nmodes) / Lz
   kz    = kz[:Nmodes]     # 10/08/2021: Why are there two definitions for kz?
   nr    = 1.0 # check later
   ωc  = ( c/nr ) * (kz**2.0 + kx**2.0)**0.5

   θ   = np.arctan(kz/kx) # check later
   #print("angle is ", θ*180.0/np.pi)

   gc0  = 0.0025/(Nmols)**0.5 # 0.1 is 100 meV
   gc   = gc0 * (ωc/ωc0)**0.5  * np.cos(θ)
   zj   = np.arange(0,Nmols) * (Lz/ Nmols)
   
   au = 27.2114
 
   nPhonon = 60
   ω0, R0 = getParameters(nPhonon,LA) 
   #print(np.sum(ω0**2*R0**2*27.2114))

   delE = (2.45- 0.09803111456132976)/au
   β = 39469.5  # 1/kBT = 1052.8 at 300 K
   #print("Beta is ",β)
   
   ndof = nPhonon *  Nmols 


def Hel(R):
    ωc = parameters.ωc
    ωm = parameters.ω0 
    gc = parameters.gc
    delE = parameters.delE
    R0 = parameters.R0
    n = parameters.NStates
    ndof = parameters.ndof
    Nmol = parameters.Nmols
    Nmodes = parameters.Nmodes
    kz = parameters.kz
    zj = parameters.zj 

    #f = parameters.factor

    Vij = np.zeros((n,n), dtype= complex)
    nPhonon = parameters.nPhonon   #ndof/Nmol  # ndof per molecule 
    # Note: In atomic units (a.u), planck constant is unity
    # diagonal terms

    # <G0| H - TR | G0>
    # ωm[nth mode]
    # for R => 1 to N nuclear modes belong to 1st molecule, 1+N to 2N modes belong to 2nd molecule
    Vij[0,0] = 0.0
    """
    for jState in  range(Nmol):
        Vij[0,0] += 0.5 * np.sum(ωm[:]**2 * R[jState * nPhonon :(jState + 1) * nPhonon]**2.0)
    """
    # R0 will have same dimention as ωm
    # R0[nth mode]
    # <Ei,0| H - TR | Ei, 0>
    for jState in range(Nmol):
        # jth mol is excited
        # so only jth * ndofpm to (jth+1)* ndofpm 
        # is shifted rest are not
        
        Vij[jState+1,jState+1] = delE #Vij[0,0]

        # 2ab term 
        # Comment 11/20/2021 Check this whether it is positive or negative. 
        # Comment 11/21/2021 Change the bottom line below
        Vij[jState+1,jState+1] -= np.sum( ωm[:]**2  * R[jState * nPhonon : (jState + 1) * nPhonon] * R0[:])
        #print(ωm[:]*27.2114)
        #print("Check R state is ", R[jState * nPhonon : (jState + 1) * nPhonon])
        #print(Vij[jState+1,jState+1])
        # b^2 term
        Vij[jState+1,jState+1] += 0.5 * np.sum( ωm[:]**2  * R0[:]**2.0)
        #print(0.5 * np.sum( ωm[:]**2  * R0[:]**2.0)*27.2114)
        #print(Vij[jState+1,jState+1])
        #Vij[jState+1,jState+1] += 0.5 * np.sum( ωm[:]**2  * R[jState * nPhonon : (jState + 1) * nPhonon] **2.0)
            
    # <G,Fi| H - TR | G,Fi>
    for fState in range(Nmodes):

        Vij[fState + 1 + Nmol, fState+1 + Nmol] =  ωc[fState] 
    

    # <G,Fi| H - TR | Ej,0>
    for fState in range(Nmodes):
        # <G,Fi| 
 
        for jState in range(Nmol):
            # | Ej,0>
            fz =  np.exp(1j * kz[fState] * zj[jState]) 
            Vij[fState + 1 + Nmol, jState + 1] = gc[fState] * fz
            Vij[jState + 1, fState + 1 + Nmol] = Vij[fState + 1 + Nmol, jState + 1].conjugate()

    return Vij


# def Hlaser(R):
#     NStates = parameters.NStates
#     E0 = parameters.LaserE
#     Phi = 0 #parameters.LaserPhi
#     NMol = parameters.NMol
#     ΔED = parameters.ΔED
#     RD = parameters.RD
#     λ = parameters.λ
#     gc = parameters.gc
#     wL = ΔED+RD**2*λ+gc*np.sqrt(NMol)
#     VMat = np.zeros((NStates,NStates))
#     for n in range(NMol):
#         VMat[0,2+n] =  E0 * np.cos(wL * t +  Phi)
#         VMat[2+n,0] =  VMat[0,2+n]

#     return VMat

def dHel0(R):
    # c = parameters.c
    ωm = parameters.ω0 
    Nmol = parameters.Nmols
    nPhonon = parameters.nPhonon#len(ωm)
    dH0 = np.zeros((len(R)))
    for jState in range(Nmol):
     dH0[jState * nPhonon : (jState + 1) * nPhonon] = ωm**2 * R[jState * nPhonon : (jState + 1) * nPhonon] 
    return dH0


def dHel(R):
    R0 = parameters.R0
    ωm = parameters.ω0
    Nmol = parameters.Nmols
    nPhonon = parameters.nPhonon
    nStates = parameters.NStates    
    dHij = np.zeros((nStates,nStates,len(R)))
    # <Ei,0| 
    for jState in range(Nmol):
        dHij[jState+1,jState+1, jState * nPhonon : (jState + 1) * nPhonon] = -ωm**2  * R0[:]
    return dHij

def initR():
    R0 = parameters.R0
    P0 = 0.0
    β  = parameters.β
    ω = parameters.ω0
    nMols = parameters.Nmols
    nPhonon = parameters.nPhonon
    ndof = parameters.ndof

    sigP = np.sqrt( ω / ( 2 * np.tanh( 0.5*β*ω ) ) )
    sigR = sigP/ω
    
    R = np.zeros(ndof)
    P = np.zeros(ndof)

    for jMol in range(nMols):
        R[jMol * nPhonon : (jMol + 1) * nPhonon] = np.random.normal(nPhonon)*sigR + 0*R0[:]
        P[jMol * nPhonon : (jMol + 1) * nPhonon] = np.random.normal(nPhonon)*sigP + P0

    return R, P