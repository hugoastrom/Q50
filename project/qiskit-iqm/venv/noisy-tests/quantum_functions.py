from qiskit.quantum_info import SparsePauliOp

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
