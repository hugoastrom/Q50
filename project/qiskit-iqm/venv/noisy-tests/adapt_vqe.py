from iqm.qiskit_iqm import IQMProvider
from iqm.qiskit_iqm.fake_backends import IQMFakeAdonis

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import PauliEvolutionGate

from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

import numpy as np
import cmath

from scipy.optimize import minimize

class QubitAdaptVQE():

    def __init__(self, ecore, h1e, h2e, optimizer):
       """
       Args:
           ecore (float): Core Hamiltonian
           h1e (np.ndarray): 1-electorn integrals
           h2e (np.ndarray): 2-electron integrals
           optimizer (string): Classical SciPy optimizer algorithm
       """

       self.ecore = ecore
       self.h1e = h1e
       self.h2e = h2e
       self.optimizer = optimizer
       self.backend = None

       self.hamiltonian = self.build_hamiltonian()
       self.nqubits = self.hamiltonian.num_qubits
       self.operator_pool = self.generate_pool()
       self.appended_ops = [SparsePauliOp(["I" * self.nqubits])]
       self.paramstring = [0.0]
       
    def set_backend(self, backend):
        self.backend = backend

    def run_qc(self, qc: QuantumCircuit, shots: int):
        if backend == None:
            raise ValueError("Backend not set!")
        
        backend = self.backend
        trans_c = transpile(qc, backend=backend)
        job = backend.run(trans_c, shots=shots)
        result = job.result()
        exp_result = job.result()._get_experiment(qc)

        return result

    def commutator(self, a: SparsePauliOp, b: SparsePauliOp):
        """
        Measure commutator AB - BA to obtain energy derivatives dE / d\theta_i = <psi|[H, O_i]|psi>
        args:
            a (SparsePauliOp): First operator
            b (SparsePauliOp): Second operator
        """

        # Prepare quantum state
        psi = self.state_prep()
        paulistring = sum(self.appended_ops)
        evolution = PauliEvolutionGate(paulistring, time=1)
        psi.compose(evolution, inplace=True)

        operator = a @ b - b @ a

        # Measure value
        estimator = Estimator()
        exp_val = estimator.run(psi, operator).result().values

        # The derivative of the energy with resoect to parameter i gets a complex phase
        exp_val = 1.0j * exp_val

        return exp_val


    def energy(self, params):
        """
        Measure energy
        """
        psi = self.state_prep()

        paulistring = SparsePauliOp([op.paulis[0] for op in self.appended_ops], coeffs = params)
        evolution = PauliEvolutionGate(paulistring, time=1)
        psi.compose(evolution, inplace=True)

        estimator = Estimator()
        e = estimator.run(psi, self.hamiltonian).result().values[0]

        return e

    def optimize_params(self, args={}):
        """
        Classically optimize parameter vector
        args:
            self.paramstring (list): Parameter vector to optimize
            args (dict): Optional arguments for the SciPy optimizer
        returns:
            Optimized parameter vector
        """

        res = minimize(energy, self.paramstring, method = self.optimizer, options=args)

        return res.x

    def generate_pool(self):
        """
        Generate operator pool
        """

        op_strings = []
        for p in range(self.nqubits - 1):
            q = self.nqubits - p - 2
            op_strings.append("I" * p + "ZY" + "I" * q)
        for p in range(self.nqubits - 1):
            q = self.nqubits - p - 1
            op_strings.append("I" * p + "Y" + "I" * q)
                
        pool = [SparsePauliOp(op) for op in op_strings]
                
        return pool

    def state_prep(self):
        """
        Initialize quantum circuit
        """
        
        psi = QuantumCircuit(self.nqubits)

        return psi

    def cholesky(self, eps):
        """
        Decompose two-body term in Hamiltonian to lower rank
        args:
            V (numpy ndarray): Two-electron integrals
            eps (float): Error resulting from decomposition
        returns:
            Numpy array representing the low-rank decomposition of the Hamiltonian two-body terms            
        """

        # see https://arxiv.org/pdf/1711.02242.pdf section B2
        # see https://arxiv.org/abs/1808.02625
        # see https://arxiv.org/abs/2104.08957
        no = self.h2e.shape[0]
        chmax, ng = 20 * no, 0
        W = self.h2e.reshape(no**2, no**2)
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
            np.abs(np.einsum("prg,qsg->prqs", L, L) - self.h2e).max(),
        )
        return L, ng

    def identity(self, n):
        """
        n-dimensional identity operator on the space of n qubits
        args:
            n (int): Number of qubits
        returns:
            SparsePauliOperator representing the identity operator
        """

        return SparsePauliOp.from_list([("I" * n, 1)])

    def qubit_mapping(self, n, mapping = "jordan_wigner"):
        """
        Map bosonic operators to fermionic operators
        args:
            n (int): Number of qubits
            mapping (str): Map bosons -> fermions
        returs:
            Lists of creation and annihilation operators
        """

        c_list = []
        if mapping == "jordan_wigner":
            for p in range(n):
                ell, r = "I" * (n - p - 1), "Z" * p
                cp = SparsePauliOp.from_list([(ell + "X" + r, 0.5), (ell + "Y" + r, 0.5j)])
                c_list.append(cp)
        else:
            raise ValueError("Unsupported mapping.")
        d_list = [cp.adjoint() for cp in c_list]
        return c_list, d_list
    
    def build_hamiltonian(self) -> SparsePauliOp:
        """
        Map one- and two-electron integrals to quantum operators
        args:
            ecore (float): Core Hamiltonian
            h1e (numpy ndarray): one-electron integrals
            h2e (numpy ndarray): two-electron integrals
        
        returns:
            Hamiltonian in fermionic operators
        """

        # Get number of orbitals
        ncas, _ = self.h1e.shape
        # List of fermionic creation and annihilation operators
        C, D = self.qubit_mapping(2 * ncas, mapping="jordan_wigner")
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
 
        # Low-rank decomposition of the Hamiltonian
        Lop, ng = self.cholesky(1e-6)
        t1e = self.h1e - 0.5 * np.einsum("pxxr->pr", self.h2e)

        # Core term
        H = self.ecore * self.identity(2 * ncas)

        # One-body term
        for p in range(ncas):
            for r in range(p, ncas):
                H += t1e[p, r] * Exc[p][r - p]

        # Two-body term
        for g in range(ng):
            Lg = 0 * self.identity(2 * ncas)
            for p in range(ncas):
                for r in range(p, ncas):
                    Lg += Lop[p, r, g] * Exc[p][r - p]
            H += 0.5 * Lg @ Lg

        return H.chop().simplify()

    def minimize_energy(self, maxiter):

        # Build Hamiltonian and generate operator pool
        
        # Data for adapt-VQE iterations
        iteration = 1
        e_last = 0
        thr = 1e-6

        
        while iteration <= maxiter:

            # List for commutators
            comm_lst = []

            # Compute commutators between the Hamiltonain and operators in pool
            for operator in self.operator_pool:
                comm_lst.append(self.commutator(self.hamiltonian, operator))

            # Find commutator with largest energy decrease and apply
            self.appended_ops.append(self.operator_pool[comm_lst.index(min(comm_lst))])
            self.paramstring.append(0.0)

            # Calculate norm of parameter vector
            #param_norm_last = np.sqrt(sum([theta ** 2 for theta in self.paramstring]))
            #self.paramstring = list(optimize_params(self.paramstring))
            #param_norm = np.sqrt(sum([theta ** 2 for theta in self.paramstring]))

            # Measure energy
            e = self.energy(self.paramstring)
            print("Appended operator is:", self.appended_ops[-1])
            print(f"Energy at iteration {iteration} is:", e)
            print("\n")

            # Converged?
            if iteration > 1:
                e_diff = e - e_last
                print("Ediff =", e_diff)
                if abs(e_diff) < thr:
                    print("Final energy:", e)
                    break

            e_last = e

            iteration += 1
