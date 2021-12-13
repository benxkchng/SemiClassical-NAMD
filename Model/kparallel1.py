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
    # Returns local acoustic phonon R0 positions and frequencies ωj
    ω = np.linspace(0.00001,0.1,10000)/27.2114
    dω = ω[1] - ω[0]
    Jω = Jω(ω)
    Fω = np.zeros((len(ω)))
    for iω in range(len(ω)):
        Fω[iω] = (4.0/np.pi) * np.sum( Jω[:iω]/ω[:iω] ) * dω
    λs = Fω[-1] 
    ωj = np.zeros((N))
    cj = np.zeros((N))
    for i in range(N):
        j = i+1
        ωj[i] = ω[np.argmin(np.abs(Fω - ((j-0.5)/N) *  λs))] 
        cj[i] = 2.0 * ωj[i] * (λs/(2.0 * N))**0.5 

    dj = cj / ωj**2
    return ωj, dj

def LA(ω):
    # Defines spectral density of local acoustic phonons
    ps = 41341.374575751
    a = 1.41586e+12
    α  =  (2.4 * ps**2.0)
    r  =  α /a
    
    ωb = 0.00223/27.2114
    return  α * (ω ** 3.0) * np.exp(-ω**2.0/(2.0 * ωb**2.0))

class parameters():
   # Simulation parameters
   NSteps = 4000*10 #int(2*10**6)
   NTraj = 1
   dtN = 1.0/10
   dtE = dtN/20
   Nmodes = 15     # Number of photonic modes
   Nmols =  7      # Number of molecules/platelets
   NStates = 1 +  Nmols + Nmodes
   M = 1
   initState = 17  # 
   # Note: We can modify code here that to allow for Franck-Condon excitation
   nskip = 100

   ωc0   = 2.4/27.2114 # Lowest cavity frequency, i.e frequency at normal incidence
   c     = 137.0 
   kx    = ωc0/c 
   # We can write Lz in terms of maximum kz value that we want; this defines a 
   # range of photon energies
   Lz    = (3.3*10**-6) * Nmols/(0.528*10**-10)
   kz    = 2 * np.pi *  np.arange(0,Nmodes) / Lz
   kz    = kz[:Nmodes]     
   nr    = 1.0       # Check with experiments
   ωc  = ( c/nr ) * (kz**2.0 + kx**2.0)**0.5

   θ   = np.arctan(kz/kx)s

   gc0  = 0.0028/(Nmols)**0.5   # This is a tunable parameter, check with experimental Rabi splitting
   gc   = gc0 * (ωc/ωc0)**0.5  * np.cos(θ)
   zj   = np.arange(0,Nmols) * (Lz/ Nmols)
   
   au = 27.2114
 
   # get Phonon parameters
   nPhonon = 50
   ω0, R0 = getParameters(nPhonon,LA) 

   delE = (2.45- 0.09803111456132976)/au # Exciton energy
   β = 39469.5  # 1/kBT = 1052.8 at 300 K
   
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

    Vij = np.zeros((n,n), dtype= complex)
    nPhonon = parameters.nPhonon   # ndof per molecule 
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
        
        Vij[jState+1,jState+1] = delE

        # 2ab term 
        Vij[jState+1,jState+1] -= np.sum( ωm[:]**2  * R[jState * nPhonon : (jState + 1) * nPhonon] * R0[:])

        # b^2 term; equals to reorganization energy
        Vij[jState+1,jState+1] += 0.5 * np.sum( ωm[:]**2  * R0[:]**2.0)
            
    # <G,Fi| H - TR | G,Fi>
    for fState in range(Nmodes):
        # Diagonal elements for photonic states
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
    ωm = parameters.ω0 
    Nmol = parameters.Nmols
    nPhonon = parameters.nPhonon
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

    # Toggle sampling between R = R0 (at excited state) and R = 0
    switch = 0.0

    # Sampling from Wigner distribution
    sigP = np.sqrt( ω / ( 2 * np.tanh( 0.5*β*ω ) ) )
    sigR = sigP/ω
    
    R = np.zeros(ndof)
    P = np.zeros(ndof)

    for jMol in range(nMols):
        R[jMol * nPhonon : (jMol + 1) * nPhonon] = np.random.normal(size=nPhonon)*sigR + switch*R0[:]
        P[jMol * nPhonon : (jMol + 1) * nPhonon] = np.random.normal(size=nPhonon)*sigP + P0

    return R, P