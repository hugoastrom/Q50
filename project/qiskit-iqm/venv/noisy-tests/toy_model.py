from iqm.qiskit_iqm import IQMProvider
from iqm.qiskit_iqm.fake_backends import IQMFakeAdonis

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import PauliEvolutionGate

from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

import numpy as np
import cmath

from scipy.optimize import minimize

def run_qc(qc, shots):
    backend = IQMFakeAdonis()
    trans_c = transpile(qc, backend=backend)
    job = backend.run(trans_c, shots=shots)
    result = job.result()
    exp_result = job.result()._get_experiment(qc)

    print(result.get_counts())

def commutator(a, b, appended_ops, paramstring):

    psi = QuantumCircuit(nqubits)
    paulistring = sum(appended_ops)
    evolution = PauliEvolutionGate(paulistring, time=1)
    psi.compose(evolution, inplace=True)

    operator = a @ b - b @ a
    
    estimator = Estimator()
    exp_val = estimator.run(psi, operator).result().values

    exp_val = 1.0j * exp_val

    return exp_val


def energy(paramstring):

    psi = QuantumCircuit(nqubits)

    paulistring = SparsePauliOp([op.paulis[0] for op in appended_ops], coeffs = paramstring)
    evolution = PauliEvolutionGate(paulistring, time=1)
    psi.compose(evolution, inplace=True)

    estimator = Estimator()
    e = estimator.run(psi, q_hamiltonian).result().values

    return e

def optimize_params(paramstring, optimizer = "nelder-mead"):

    theta = paramstring
    res = minimize(energy, theta, method = optimizer, options={'xatol': 1e-8, 'disp': True})
    
    return res.x

if __name__ == "__main__":

    operator_pool = [SparsePauliOp(["ZY"]), SparsePauliOp(["YI"])]
    q_hamiltonian = SparsePauliOp(["IZ", "ZI"], coeffs = [1 / 2, 1 / 2])

    nqubits = 2
    shots = 1000
    max_iter = 10

    iteration = 1
    appended_ops = [SparsePauliOp(["I" * nqubits])]
    paramstring = [0.0]
    e = energy(paramstring)[0]
    print("Initial energy:", e)

    e_last = 0

    thr = 1e-6

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
            print("Final energy:", e)
            break


        # Measure energy
        print("Appended operator is:", appended_ops[-1], "with parameter theta =", paramstring[-1])
        print(f"Energy at iteration {iteration} is:", e)
        print("\n")

        # Converged?
        #comm_norm = np.sqrt(sum([comm * np.conjugate(comm) for comm in comm_lst]))
        if iteration > 1:
            e_diff = e - e_last
            print("Ediff =", e_diff)
            if abs(e_diff) < thr:
                print("Final energy:", e)
                break
            
        e_last = e

        iteration += 1
