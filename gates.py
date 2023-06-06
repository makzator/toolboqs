import numpy as np


def _idx_to_dits(i, nstates):
    # nstates = [n1, n2, n3...], |i> = |a1, a2, a3, ...>
    a = []
    idx = i
    for n in range(len(nstates)):
        a.append(idx % nstates[-1-n])
        idx = idx // nstates[-1-n]
    return a[::-1]

def _dits_to_idx(dits, nstates):
    inv_nstates_cumprod = np.ones(len(nstates))
    inv_nstates_cumprod[1:] = np.cumprod(nstates[:0:-1])
    return int(np.multiply(dits, inv_nstates_cumprod[::-1]).sum())

def extend(gate, ness, nguards):
    # ness, nguards are lists [ng1, ng2, ...]
    gate = np.array(gate)
    dim_ess = np.prod(ness)
    assert gate.shape == (dim_ess,dim_ess), "gate not compatible w ness"

    ntot = np.array(ness) + np.array(nguards)
    dimtot = ntot.prod()
    gate_full = np.zeros((dimtot,dimtot), dtype=complex)

    idx_dits = [_idx_to_dits(i, ness) for i in range(dim_ess)]
    idx_mapping = [_dits_to_idx(dits, ntot) for dits in idx_dits]

    for i in range(dim_ess):
        for j in range(dim_ess):
            gate_full[idx_mapping[i],idx_mapping[j]] = gate[i,j]
    return gate_full

def gate_rows(gate):
    gate = np.round(gate, decimals=10)
    return gate.real.reshape(gate.shape[0]*gate.shape[1]), gate.imag.reshape(gate.shape[0]*gate.shape[1])

def swap_order(gate, dimA, dimB):
    """
    Compute gate matrix for swapped order of objects in two-object gate
    |AB> -> |BA>
    """
    gate_new = np.zeros_like(gate)
    for i in range(gate.shape[0]):
        ai = i // dimB
        bi = i % dimB
        i_new = bi * dimA + ai
        for j in range(gate.shape[1]):
            aj = j // dimB
            bj = j % dimB
            j_new = bj * dimA + aj
            gate_new[i_new,j_new] = gate[i,j]
    return gate_new

sigmax = np.array([[0, 1], [1, 0]])
def ENC_1(pow=1):
    alpha = np.pi/2 * pow
    sigmax_pow = np.exp(-1j*alpha) * (np.cos(alpha)*np.identity(2) + 1j*np.sin(alpha)*sigmax)
    A = np.kron(sigmax_pow, np.identity(2))
    ENC = np.identity(8, dtype=complex)
    ENC[2:6,2:6] = A
    ENC = np.round(ENC, decimals=12)
    return ENC

def DEC_1(pow=1):
    ENC = ENC_1(pow)
    return ENC.conjugate().T

def SWAPd(d=2, pow=1):
    """
    Qudit SWAP
    |01> <-> |10>...
    """
    # swap = np.zeros((d**2, d**2))
    # for i in range(d):
    #     for j in range(d):
    #         state1 = i*d + j    # |ij>
    #         state2 = j*d + i    # |ji>
    #         swap[state1, state2] = 1
    #         swap[state2, state1] = 1

    Lambda = np.identity(d*d, dtype=complex)
    Lambda[-d*(d-1)//2:] *= np.exp(1j * np.pi * pow)
    U = np.zeros((d*d, d*d))
    for i in range(d):
        U[(d+1)*i, i] = 1
    i = 0
    for a in range(d):
        for b in range(a+1,d):
            U[a*d+b,d+i] = 1/np.sqrt(2)
            U[b*d+a,d+i] = 1/np.sqrt(2)
            U[a*d+b,d*(d+1)//2+i] = 1/np.sqrt(2)
            U[b*d+a,d*(d+1)//2+i] = -1/np.sqrt(2)
            i += 1
    swap = U @ Lambda @ U.T
    return swap

def pSWAPup():
    return ENC_1()

def pSWAPdown():
    """
    qubit-ququart partial SWAP (C) (A B) -> (B) (A C)
    |00> <-> |00>
    |01> <-> |10>
    |02> <-> |02>
    |03> <-> |12>
    |11> <-> |11>
    |13> <-> |13>
    """
    return np.array([
        [1,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,1,0],
        [0,1,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,1]
    ])

def p4SWAPdown():
    """
    ququart-ququart partial SWAP down |BA> |DC> -> |BC> |DA>
    """
    swaps = [(1,4), (3,6), (9,12), (11,14)]
    gate = np.eye(16)
    for swap in swaps:
        a, b = swap
        gate[a,a] = 0
        gate[a,b] = 1
        gate[b,b] = 0
        gate[b,a] = 1
    return gate

def qqSWAP(QQ):
    """
    ququart-ququart SWAP |BA> |DC>
    QQ: two-character string indicating involved qubits, e.g. "AD"
    """
    def base4(str):
        return 4*int(str[0]) + int(str[1])

    if QQ == "BD":
        trans = [["02","20"], ["03","21"], ["12","30"], ["13","31"]]
    elif QQ == "AD":
        trans = [["02","10"], ["03","11"], ["22","30"], ["23","31"]]
    elif QQ == "BC":
        trans = [["01","20"], ["03","22"], ["11","30"], ["13","32"]]
    elif QQ == "AC":
        trans = [["01","10"], ["03","12"], ["21","30"], ["23","32"]]
    else:
        raise Exception('Wrong QQ key. Possible: "AC", "AD", "BC", "BD".')

    gate = np.eye(16)
    for t in trans:
        a = base4(t[0])
        b = base4(t[1])
        gate[a,a] = 0
        gate[a,b] = 1
        gate[b,b] = 0
        gate[b,a] = 1
    return gate

def dCNOT(d, c):
    """
    qudit-qubit CNOT, control is on qudit state |c> 
    """
    gate = np.eye(d*2)
    gate[[2*c, 2*c+1], 2*c] = [0,1]
    gate[[2*c, 2*c+1], 2*c+1] = [1,0]
    return gate

def qqCNOT(CT):
    """
    ququart-ququart CNOT |BA> |DC>
    CT: two-character string indicating control and target, e.g. "AD"
    control is in first qq and target in second qq
    """
    def base4(str):
        return 4*int(str[0]) + int(str[1])

    if CT == "BD":
        trans = [["20","22"], ["21","23"], ["30","32"], ["31","33"]]
    elif CT == "AD":
        trans = [["10","12"], ["11","13"], ["30","32"], ["31","33"]]
    elif CT == "BC":
        trans = [["20","21"], ["22","23"], ["30","31"], ["32","33"]]
    elif CT == "AC":
        trans = [["10","11"], ["12","13"], ["30","31"], ["32","33"]]
    else:
        raise Exception('Wrong CT key. Possible: "AC", "AD", "BC", "BD".')

    gate = np.eye(16)
    for t in trans:
        a = base4(t[0])
        b = base4(t[1])
        gate[a,a] = 0
        gate[a,b] = 1
        gate[b,b] = 0
        gate[b,a] = 1
    return gate

def Hd(d=2):
    """
    Generalized Hadamard H_d
    (H_d)_ij = w_ij = exp(2 pi ij/d)
    """
    m = np.arange(d)
    return np.exp(2j*np.pi/d * np.outer(m,m)) / np.sqrt(d)

def Td(d=2):
    """
    Generalized T T_d
    """
    return np.diag(np.exp(2j*np.pi/d * np.arange(d)/4))

def _frac_mod(a, b, p):
    n = 0
    while n*b % p != a % p:
        n += 1
    return n

def Td_2(d=2, zprime=0, gprime=0, eprime=0):
    """
    Generalized U_(pi/8) as in https://journals.aps.org/pra/pdf/10.1103/PhysRevA.86.022316
    """
    if d == 2:
        return Td()
    elif d == 3:
        v = np.array([
            0,
            6*zprime + 2*gprime + 3*eprime,
            6*zprime + 1*gprime + 6*eprime
        ]) % 9
        return np.diag(np.exp(2j*np.pi/9 * v))
    else:
        k = np.arange(d)
        v = k * eprime
        tmp = k * (gprime + k*(6*zprime + (2*k - 3)*gprime))
        for _k in range(d):
            tmp[_k] = _frac_mod(tmp[_k], 12, d)
        v += tmp
        return np.diag(np.exp(2j*np.pi/d * v))

def X1d(d=2):
    """
    X_+1 gate
    X_+1 |i> = |(i+1) mod d>
    """
    x1d = np.diag(np.ones(d-1),k=-1)
    x1d[0,d-1] = 1
    return x1d

def SWAP0d(d=2):
    """
    swap |0> and |d-1> in qudit
    """
    swap0d = np.identity(d)
    swap0d[0,0] = 0
    swap0d[0,d-1] = 1
    swap0d[d-1,0] = 1
    swap0d[d-1,d-1] = 0
    return swap0d
