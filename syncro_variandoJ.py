from qutip import *
import numpy as np
from scipy.sparse import csc_matrix
from scipy.linalg import expm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
import warnings
import multiprocessing as mp


def US(ts, HS1, HS2): #Time - evolution operator for free dynamics of Spin 1 and Spin 2
    mat = expm(-1j * (HS1 + HS2) * ts)
    return csc_matrix(mat)

def US1S2(tss, HS1S2): #Time - evolution operator for Spin 1 and Spin 2 interaction
    mat = expm(-1j * HS1S2 * tss)
    return csc_matrix(mat)

def US2E(tse, HS2E): #Time - evolution operator for Spin 2 and environment
    mat = expm(-1j * HS2E * tse)
    return csc_matrix(mat)

def pearson_corr_expec(w1, lam, J, cn):
    ts = 1
    tss = 1
    tse = 1
    warnings.filterwarnings("ignore")
    w2 = 0.2
    HS1 = (-w1/2)*tensor(sigmaz(), qeye(2), qeye(2)) #Self Hamiltonian of Spin 1
    HS2 = (-w2/2)*tensor(qeye(2), sigmaz(), qeye(2)) #Self Hamiltonian of Spin 2
    HS1S2 = lam/2*(tensor(sigmax(), sigmax(), qeye(2)) + tensor(sigmay(), sigmay(), qeye(2))) #Interaction Hamiltonian for Spin 1 and Spin 2
    HS2E = (1/2)*(J*tensor(qeye(2), sigmax(), sigmax())+ J * tensor(qeye(2), sigmay(), sigmay()))

    psi_S_initial = csc_matrix(np.kron(np.array([[np.cos(np.pi/4)], [np.sin(np.pi/4)]]), np.array([[np.cos(np.pi/4)], [np.sin(np.pi/4)]])))
    rho_S_initial = psi_S_initial @ psi_S_initial.getH()
    rho_E_initial = Qobj(np.array([[1., 0.], [0., 0.]]))
    rho_0 = rho_S_initial

    rho_S1 = [csc_matrix(np.zeros((2, 2))) for _ in range(cn + 1)]
    rho_S2 = [csc_matrix(np.zeros((2, 2))) for _ in range(cn + 1)]
    rho_S1S2 = [csc_matrix(np.zeros((4, 4))) for _ in range(cn + 1)]
    rho = [csc_matrix(np.zeros((4, 4))) for _ in range(cn + 1)]
    

    rho_S1[0] = Qobj(rho_S_initial, dims=[[2, 2], [2, 2]]).ptrace(0)
    rho_S2[0] = Qobj(rho_S_initial, dims=[[2, 2], [2, 2]]).ptrace(1)
    rho_S1S2[0] = rho_S_initial
    rho[0] = rho_0

    for ii in range(0, cn):
        rho_rho = csc_matrix(tensor(Qobj(rho[ii]), rho_E_initial))
        rho_C = US(ts, HS1, HS2) @ US1S2(tss, HS1S2) @ US2E(tse, HS2E) @ rho_rho @ US2E(tse, HS2E).conj().T @ US1S2(tss, HS1S2).conj().T @ US(ts, HS1, HS2).conj().T
        rho_Sys1_Sys2 = Qobj(rho_C, dims=[[2,2,2], [2,2,2]]).ptrace([0,1])
        rho_Sys1 = Qobj(rho_C, dims=[[2,2,2], [2,2,2]]).ptrace(0)
        rho_Sys2 = Qobj(rho_C, dims=[[2,2,2], [2,2,2]]).ptrace(1)
        rho_Sys_Env = Qobj(rho_C, dims=[[2,2,2], [2,2,2]]).ptrace([0,1])
        rho_S1[ii+1] = (1/rho_Sys1.tr())*rho_Sys1
        rho_S2[ii+1] = (1/rho_Sys2.tr())*rho_Sys2
        rho_S1S2[ii+1] = (1/rho_Sys1_Sys2.tr())*rho_Sys1_Sys2
        rho[ii+1] = (1/rho_Sys_Env.tr())*rho_Sys_Env

    # Calculate the X expectation values for spin 1 and spin 2
    expecXspin1 = np.array([np.trace(rho_S1[kk] * sigmax()).astype(float) for kk in range(0, cn)])
    expecXspin2 = np.array([np.trace(rho_S2[jj] * sigmax()).astype(float) for jj in range(0, cn)])
    
    return expecXspin1, expecXspin2



