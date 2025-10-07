from iqm.qiskit_iqm import IQMProvider
from iqm.qiskit_iqm.fake_backends import IQMFakeAdonis

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import XGate, ZGate

from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

import numpy as np
import cmath

def run_qc(qc, shots):
    backend = IQMFakeAdonis()
    trans_c = transpile(qc, backend=backend)
    job = backend.run(trans_c, shots=shots)
    result = job.result()
    exp_result = job.result()._get_experiment(qc)

    print(result.get_counts())

def commutator(a, b):

    psi = QuantumCircuit(nqubits)
    #psi.h(0)

    operator = a @ b - b @ a
    
    for op in appended_ops:
        operator = op.adjoint() @ operator @ op

    #print(operator)
    estimator = Estimator()
    exp_val = estimator.run(psi, operator).result().values#.real

    print("Expectation value:", exp_val)

    return exp_val

if __name__ == "__main__":

    operator_pool = [SparsePauliOp(["ZY"], coeffs = [complex(0.0, 1.0)]), SparsePauliOp(["YI"], coeffs = [complex(0.0, 1.0)])]
    #qubit_hamiltonian = mapped(fermion_hamiltonian) # From LUCAS/PySCF?
    q_hamiltonian = SparsePauliOp(["IZ", "ZI"], coeffs = [1 / 2, 1 / 2])
    #ansatz = mapped(HF_state) # From LUCAS/PySCF?

    nqubits = 2
    shots = 1000
    max_iter = 10

    iteration = 1
    appended_ops = []

    e_last = 0

    thr = 1e-3

    while iteration < max_iter:

        comm_lst = []

        for operator in operator_pool:
            comm_lst.append(commutator(operator, q_hamiltonian))

        # Find largest commutator and apply
        appended_ops.append(operator_pool[comm_lst.index(min(comm_lst))])

        # Measure energy
        psi = QuantumCircuit(nqubits)
        #psi.h(0)
        operator = SparsePauliOp([nqubits * "I"])
        for op in appended_ops:
            operator = op @ operator

        operator = operator.adjoint() @ q_hamiltonian @ operator
        estimator = Estimator()
        energy = estimator.run(psi, operator).result().values#.real
        print(f"Energy at iteration {iteration} is:", energy[0])
        print("Appended operator is:", appended_ops[-1])


        # Converged?
        if iteration > 1:
            e_diff = abs(energy - e_last)
            if e_diff < thr:
                print("Final energy:", energy[0])
                break

        e_last = energy

        iteration += 1


        # Classically optimize parameters
