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

def reduce_pauli_op(op):

    d = {"XX": ("", 1.0), "XY": ("Z", complex(0.0, 1.0)), "XZ": ("Y", complex(0.0, -1.0)), "YX": ("Z", complex(0.0, -1.0)), "YY": ("", -1.0), "YZ": ("X", complex(0.0, 1.0)), "ZX": ("Y", complex(0.0, 1.0)), "ZY": ("X", complex(0.0, -1.0)), "ZZ": ("", 1.0)}
    op[1] *= d[op[0][:2]][1]
    op[0] = op[0].replace(op[0][:2], d[op[0][:2]][0])

    return
    
def commutator(a, b):

    psi = QuantumCircuit(nqubits)
    for op in appended_ops:
        psi.append(op[0], [op[1]])

    ab = [["", 1.0] for qubit in range(nqubits)]
    for pauliop in a:
        ab[pauliop[1]][0] += pauliop[0]
    for pauliop in b:
        ab[pauliop[1]][0] += pauliop[0]
    for op in ab:
        while len(op[0]) > 1:
            reduce_pauli_op(op)

    ba = [["", 1.0] for qubit in range(nqubits)]
    for pauliop in b:
        ba[pauliop[1]][0] += pauliop[0]
    for pauliop in a:
        ba[pauliop[1]][0] += pauliop[0]
    for op in ba:
        while len(op[0]) > 1:
            reduce_pauli_op(op)

    first_op = ''
    first_coeff = 1.0
    for qubit in ab:
        first_op += "I" if len(qubit[0]) == 0 else qubit[0]
        first_coeff *= qubit[1]
        
    second_op = ''
    second_coeff = 1.0
    for qubit in ba:
        second_op += "I" if len(qubit[0]) == 0 else qubit[0]
        second_coeff *= qubit[1]

    comm = SparsePauliOp([first_op, second_op], coeffs=[first_coeff, -1.0 * second_coeff])
    print(comm)
    estimator = Estimator()

    exp_val = estimator.run(psi, comm).result().values#.real

    print("Expectation value:", exp_val)
    
    return exp_val

def apply_gate(qc, op, qbit):

    qc.append(op, [qbit])

if __name__ == "__main__":
    
    #operator_pool = list[QuantumCurcuit]
    #operator_pool = [("x", 0), ("z", 0), ("y", 1)]
    #operator_pool = [(XGate(), 0), (ZGate(), 1)] #GlobalPhaseGate(math.pi / 2) is a global phase of i
    operator_pool = [[("X", 0)], [("Z", 1)]]
    #qubit_hamiltonian = mapped(fermion_hamiltonian) # From LUCAS/PySCF?
    #q_hamiltonian = ("Z", 0)
    q_hamiltonian = [("Z", 0)]
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
            psi.append(op[0], op[1])

        estimator = Estimator()
        hamiltonian = SparsePauliOperator([pauli_term[0] for pauli_term in q_hamiltoian], coeffs = [1.0 for pauli_term in q_hamiltonian])
        energy = estimator.run(psi, hamiltonian).result().values#.real



        e_diff = e_last - energy
        
        # Energy difference < energy_threshold?
        if e_diff < thr:
            break
        
        e_last = energy
        
        # Classically optimize parameters
