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
    for op in appended_ops:
        psi.append(op)

    comm = a @ b - b @ a
    estimator = Estimator()

    exp_val = estimator.run(psi, comm).result().values#.real

    print("Expectation value:", exp_val)

    return exp_val

if __name__ == "__main__":

    #operator_pool = [[("X", 0)], [("Z", 1)]]
    operator_pool = [SparsePauliOp(["YZ"], coeffs = [complex(0.0, 1.0)]), SparsePauliOp(["IY"], coeffs = [complex(0.0, 1.0)])]
    #qubit_hamiltonian = mapped(fermion_hamiltonian) # From LUCAS/PySCF?
    #q_hamiltonian = ("Z", 0)
    q_hamiltonian = SparsePauliOp(["ZI"])
    #ansatz = mapped(HF_state) # From LUCAS/PySCF?


    nqubits = 2
    shots = 1000
    max_iter = 10

    iteration = 0
    appended_ops = []

    while iteration < max_iter:

        comm_lst = []

        for operator in operator_pool:
            comm_lst.append(commutator(operator, q_hamiltonian))

        # Find largest commutator and apply
        appended_ops.append(operator_pool[comm_lst.index(min(comm_lst))])

        # Measure energy
        psi = QuantumCircuit()
        for op in appended_ops:
            psi.append(op)

        estimator = Estimator()
        energy = estimator.run(psi, q_hamiltonian).result().values#.real
        print(energy)

        e_diff = e_last - energy

        # Energy difference < energy_threshold?
        if e_diff < thr:
            break

        e_last = energy

        # Classically optimize parameters
