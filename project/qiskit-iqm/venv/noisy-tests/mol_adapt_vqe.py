from iqm.qiskit_iqm import IQMProvider
from iqm.qiskit_iqm.fake_backends import IQMFakeAdonis

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import PauliEvolutionGate

from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

import numpy as np
import cmath

from scipy.optimize import minimize

from pyscf import ao2mo, gto, mcscf, scf

def cholesky(V, eps):
    # see https://arxiv.org/pdf/1711.02242.pdf section B2
    # see https://arxiv.org/abs/1808.02625
    # see https://arxiv.org/abs/2104.08957
    no = V.shape[0]
    chmax, ng = 20 * no, 0
    W = V.reshape(no**2, no**2)
    L = np.zeros((no**2, chmax))
    Dmax = np.diagonal(W).copy()
    nu_max = np.argmax(Dmax)
    vmax = Dmax[nu_max]
    while vmax > eps:
        L[:, ng] = W[:, nu_max]
        if ng > 0:
            L[:, ng] -= np.dot(L[:, 0:ng], (L.T)[0:ng, nu_max])
        L[:, ng] /= np.sqrt(vmax)
        Dmax[: no**2] -= L[: no**2, ng] ** 2
        ng += 1
        nu_max = np.argmax(Dmax)
        vmax = Dmax[nu_max]
    L = L[:, :ng].reshape((no, no, ng))
    print(
        "accuracy of Cholesky decomposition ",
        np.abs(np.einsum("prg,qsg->prqs", L, L) - V).max(),
    )
    return L, ng

def identity(n):
    return SparsePauliOp.from_list([("I" * n, 1)])

def qubit_mapping(n, mapping = "jordan_wigner"):
    c_list = []
    if mapping == "jordan_wigner":
        for p in range(n):
            if p == 0:
                ell, r = "I" * (n - 1), ""
            elif p == n - 1:
                ell, r = "", "Z" * (n - 1)
            else:
                ell, r = "I" * (n - p - 1), "Z" * p
            cp = SparsePauliOp.from_list([(ell + "X" + r, 0.5), (ell + "Y" + r, 0.5j)])
            c_list.append(cp)
    else:
        raise ValueError("Unsupported mapping.")
    d_list = [cp.adjoint() for cp in c_list]
    return c_list, d_list

def build_hamiltonian(ecore: float, h1e: np.ndarray, h2e: np.ndarray) -> SparsePauliOp:
    ncas, _ = h1e.shape
 
    C, D = qubit_mapping(2 * ncas, mapping="jordan_wigner")
    Exc = []
    for p in range(ncas):
        Excp = [C[p] @ D[p] + C[ncas + p] @ D[ncas + p]]
        for r in range(p + 1, ncas):
            Excp.append(
                C[p] @ D[r]
                + C[ncas + p] @ D[ncas + r]
                + C[r] @ D[p]
                + C[ncas + r] @ D[ncas + p]
            )
        Exc.append(Excp)
 
    # low-rank decomposition of the Hamiltonian
    Lop, ng = cholesky(h2e, 1e-6)
    t1e = h1e - 0.5 * np.einsum("pxxr->pr", h2e)
 
    H = ecore * identity(2 * ncas)
    # one-body term
    for p in range(ncas):
        for r in range(p, ncas):
            H += t1e[p, r] * Exc[p][r - p]
    # two-body term
    for g in range(ng):
        Lg = 0 * identity(2 * ncas)
        for p in range(ncas):
            for r in range(p, ncas):
                Lg += Lop[p, r, g] * Exc[p][r - p]
        H += 0.5 * Lg @ Lg
 
    return H.chop().simplify()

def run_qc(qc, shots):
    backend = IQMFakeAdonis()
    trans_c = transpile(qc, backend=backend)
    job = backend.run(trans_c, shots=shots)
    result = job.result()
    exp_result = job.result()._get_experiment(qc)

    print(result.get_counts())

def commutator(a, b, appended_ops, paramstring):

    #psi = QuantumCircuit(nqubits)
    psi = prepare_ansatz(nqubits)
    paulistring = sum(appended_ops)
    evolution = PauliEvolutionGate(paulistring, time=1)
    psi.compose(evolution, inplace=True)

    operator = a @ b - b @ a
    
    estimator = Estimator()
    exp_val = estimator.run(psi, operator).result().values

    exp_val = 1.0j * exp_val
    print("\nOperator:", b, "Commutator:", exp_val, "\n")

    return exp_val


def energy(paramstring):

    #psi = QuantumCircuit(nqubits)
    psi = prepare_ansatz(nqubits)

    paulistring = SparsePauliOp([op.paulis[0] for op in appended_ops], coeffs = paramstring)
    evolution = PauliEvolutionGate(paulistring, time=1)
    psi.compose(evolution, inplace=True)

    estimator = Estimator()
    e = estimator.run(psi, q_hamiltonian).result().values

    return e

def optimize_params(paramstring):

    theta = paramstring
    res = minimize(energy, theta, method = "nelder-mead", options={'xatol': 1e-8, 'disp': True})

    return res.x

def prepare_ansatz(nqubits):

    ansatz = QuantumCircuit(nqubits)
    #ansatz.x(0)
    #ansatz.x(1)
    #evolution = PauliEvolutionGate(q_hamiltonian, time=1)
    #ansatz.compose(evolution, inplace=True)

    return ansatz

if __name__ == "__main__":

    #a = 0.735 / 2
    a = 1.5
    
    mol = gto.Mole()
    mol.build(
        verbose = 0,
        atom = [["He", (0, 0, 0)]],#, ["H", (0, 0, a)]],
        basis = "sto-3g",
        spin = 0,
        charge = 0,
        #symmetry = "Dooh"
    )

    mf = scf.RHF(mol)
    mf.scf()
    
    print("HF electronic energy:", mf.energy_elec()[0])

    #ecore = mf.get_hcore()
    #h1e = mf.get_ovlp()
    #h2e = mol.intor("int2e", aosym="s8")
    
    #active_space = range(mol.nelectron // 2 - 1, mol.nelectron // 2 + 1)
    active_space = range(10)

    E1 = mf.kernel()
    mx = mcscf.CASCI(mf, ncas=10, nelecas=(1, 1))
    #mo = mx.sort_mo(active_space, base=0)
    #E2 = mx.kernel(mo)[:2]
    
    h1e, ecore = mx.get_h1eff()
    #h2e = ao2mo.restore(1, mx.get_h2eff(), mx.ncas)
    h2e = mx.get_h2eff()

    print(ecore)
    print(h1e)
    print(h2e)


    #operator_pool = [SparsePauliOp(["IIZY"]), SparsePauliOp(["IIYI"]), SparsePauliOp(["IZYI"]), SparsePauliOp(["IYII"]), SparsePauliOp(["ZYII"]), SparsePauliOp(["YIII"])]
    operator_pool = [SparsePauliOp(["ZY"]), SparsePauliOp(["YI"])]
    q_hamiltonian = build_hamiltonian(ecore, h1e, h2e)

    print(q_hamiltonian)
    
    nqubits = 2
    shots = 1000
    max_iter = 100

    iteration = 1
    appended_ops = [identity(nqubits)]

    paramstring = [0.0]

    e_last = 0

    thr = 1e-6

    psi = prepare_ansatz(nqubits)
    estimator = Estimator()
    e = estimator.run(psi, q_hamiltonian).result().values
    print("Initial energy:", e)

    while iteration <= max_iter:

        comm_lst = []

        for operator in operator_pool:
            comm_lst.append(commutator(q_hamiltonian, operator, appended_ops, paramstring))

        # Find largest commutator and apply
        appended_ops.append(operator_pool[comm_lst.index(min(comm_lst))])
        paramstring.append(0.0)

        param_norm_last = np.sqrt(sum([theta ** 2 for theta in paramstring]))
        paramstring = list(optimize_params(paramstring))
        param_norm = np.sqrt(sum([theta ** 2 for theta in paramstring]))

        e = energy(paramstring)[0]

        print("Change in norm is:", param_norm_last - param_norm)
        if abs(param_norm_last - param_norm) < thr:
            print("Initial HF energy:", mf.energy_tot())
            print("Final energy:", e)
            break


        # Measure energy
        print("Appended operator is:", appended_ops[-1], "with coefficient: ", paramstring[-1])
        print(f"Energy at iteration {iteration} is:", e)
        print("\n")

        #comm_norm = np.sqrt(sum([comm ** 2 for comm in comm_lst]))
        
        # Converged?
        if iteration > 1:
            e_diff = e - e_last
            print("Ediff =", e_diff)
            if abs(e_diff) < thr:
                print("Initial HF energy:", mf.energy_tot())
                print("Final energy:", e)
                break
            
        e_last = e

        iteration += 1
