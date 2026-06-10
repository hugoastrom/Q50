from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit

import numpy as np
from itertools import product

def qubit_mapping(n, mapping = "jordan_wigner"):
    """
    Map bosonic operators to fermionic operators
    args:
        n (int): Number of qubits
        mapping (str): Map bosons -> fermions
    returns:
        Lists of creation and annihilation operators
    """

    c_list = []
    if mapping == "jordan_wigner":
        for p in range(n):
            z_string = "Z" * p
            x_string = z_string + "X" + "I" * (n - p - 1)
            y_string = z_string + "Y" + "I" * (n - p - 1)
            cp = SparsePauliOp.from_list([
                (x_string, 0.5),
                (y_string, -0.5j)
            ])
            c_list.append(cp)
    else:
        raise ValueError("Unsupported mapping.")
    d_list = [cp.adjoint() for cp in c_list]
    return c_list, d_list

def identity(n):
        """
        n-dimensional identity operator on the space of n qubits
        args:
            n (int): Number of qubits
        returns:
            SparsePauliOperator representing the identity operator
        """
        return SparsePauliOp.from_list([("I" * n, 1)])

# Error mitigation functions
def calibration_circuits(n):
    circuits = []
    labels = []

    for bits in product([0, 1], repeat=n):
        qc = QuantumCircuit(n, n)

        for i, b in enumerate(bits):
            if b == 1:
                qc.x(i)

        qc.measure(range(n), range(n))

        circuits.append(qc)
        labels.append("".join(map(str, bits)))

    return circuits, labels

def build_confusion_matrix(nqubits, labels, results):

    dim = 2 ** nqubits
    M = np.zeros((dim, dim))

    label_to_index = {label: i for i, label in enumerate(labels)}
        
    for i, label in enumerate(labels):
        counts = results.get_counts(i)
        shots = sum(counts.values())
            
        for bitstring, count in counts.items():
            bitstring = bitstring[::-1]
            j = label_to_index[bitstring]
            M[j, i] += count / shots

    M_inv = np.linalg.pinv(M)

    return M_inv


def mitigate_counts(counts, labels, nqubits, M_inv):
    dim = len(labels)
    vec = np.zeros(dim)
    
    label_to_index = {label: i for i, label in enumerate(labels)}

    shots = sum(counts.values())

    for bitstring, count in counts.items():
        bitstring = bitstring[::-1]
        idx = label_to_index[bitstring]
        vec[idx] = count / shots
    
    mitigated = M_inv @ vec

    return mitigated
