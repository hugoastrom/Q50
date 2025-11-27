
# Qiskit IQM packages
from iqm.qiskit_iqm import IQMProvider
from iqm.qiskit_iqm.fake_backends import IQMFakeAdonis

# Qiskit packages
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate, StatePreparation
from qiskit.primitives import Estimator, StatevectorEstimator, BackendEstimatorV2
from qiskit.quantum_info import SparsePauliOp

# Python packages
import numpy as np
import cmath
from scipy.optimize import minimize

class QubitAdaptVQE():
    
    def __init__(self, mo_occs, h1e, h2e, optimizer):
        """
        Args:
            mo_occs (list): MO occupations
            h1e (np.ndarray): 1-electron integrals
            h2e (np.ndarray): 2-electron integrals
            optimizer (string): Classical SciPy optimizer algorithm
        """
        
        # Read input
        norb = h1e.shape[0]
        # Ansatz is ON vector for 2 * norb spin orbitals
        self.ansatz = []
        for iocc, occ in enumerate(mo_occs):
            if occ > 0:
                self.ansatz.append(iocc)
                if occ > 1:
                    self.ansatz.append(iocc + norb)
        self.h1e = h1e
        self.h2e = h2e
        self.optimizer = optimizer
        
        # Declare variables needed
        self.backend = None
        self.use_cholesky = False

        # Build Hamiltonian
        self.hamiltonian = self.build_hamiltonian()
        print(self.hamiltonian)
        self.nqubits = self.hamiltonian.num_qubits
        print("Number of qubits = %i\n" %self.nqubits)

        # Construct operator pool and parameters
        self.operator_pool = self.generate_pool()
        self.appended_ops = [SparsePauliOp(["I" * self.nqubits])]
        self.paramstring = [0.0]

    def estimator_dict(self, estimator):
        d = {"estimator": Estimator(),
             "backend_estimator": BackendEstimatorV2(backend = self.backend),
             "statevector_estimator": StatevectorEstimator()
             }
        return d[estimator]

    def set_backend(self, backend):
        self.backend = backend

    def run_qc(self, qc: QuantumCircuit, shots: int):

        trans_c = transpile(qc, backend=self.backend)
        job = backend.run(trans_c, shots=shots)
        result = job.result()
        exp_result = job.result()._get_experiment(qc)

        return result

    def calc_exp_val(self, qc: QuantumCircuit, op: SparsePauliOp):
        """
        Calculate expectation value of op using qc
        args:
            qc (QuntumCircuit): The state
            op (SparsePauliOp): The observable
        """
        estimator = self.estimator
        if isinstance(estimator, Estimator):
            res = estimator.run(qc, op).result().values
        elif isinstance(estimator, StatevectorEstimator):
            res = estimator.run([(qc, op)]).result()[0].data.evs
        elif isinstance(estimator, BackendEstimatorV2):
            qc = transpile(qc, backend=self.backend)
            res = estimator.run([(qc, op)]).result()[0].data.evs
        else:
            raise ValueError("Undefined estimator")

        return res

    def set_estimator(self, estimator):
        """
        Set the classical estimator
        options:
            estimator
            backend_estimator
            statevector_estimator
        """
        try:
            self.estimator = self.estimator_dict(estimator)
        except KeyError:
            print("Estimator not implemented")
        if estimator == "backend_estimator" and self.backend == None:
            raise ValueError("Set backend!")

        return

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
        # The derivative of the energy with respect to parameters yields a complex phase
        operator = 1.0j * operator.chop().simplify()

        # Measure value
        exp_val = self.calc_exp_val(psi, operator)

        return exp_val

    def energy(self, params):
        """
        Measure energy
        """
        psi = self.state_prep()
        # Need to declare the parameters explicitly for SciPy optimization routine
        paulistring = SparsePauliOp([op.paulis[0] for op in self.appended_ops], coeffs = params)
        evolution = PauliEvolutionGate(paulistring, time=1)
        psi.compose(evolution, inplace=True)

        e = self.calc_exp_val(psi, self.hamiltonian)

        return e

    def optimize_params(self, args={"disp": True}):
        """
        Classically optimize parameter vector
        args:
            args (dict): Optional arguments for the SciPy optimizer
        returns:
            Optimized parameter vector
        """

        res = minimize(self.energy, self.paramstring, method = self.optimizer, options=args)

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
        for iocc in self.ansatz:
            psi.x(self.nqubits - iocc - 1)

        return psi

    def cholesky(self, eps: float):
        """
        Decompose two-body term in Hamiltonian to lower rank
        args:
            eps (float): Error threshold
        returns:
            Numpy array representing the low-rank decomposition of the Hamiltonian two-body term
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
                z_op =  "Z" * p
                id_op = "I" * (n - p - 1)
                cp = SparsePauliOp.from_list([(id_op + "X" + z_op, 0.5), (id_op + "Y" + z_op, 0.5j)])
                c_list.append(cp)
        else:
            raise ValueError("Unsupported mapping.")
        d_list = [cp.adjoint() for cp in c_list]
        return c_list, d_list

    def build_hamiltonian(self) -> SparsePauliOp:
        """
        Map one- and two-electron integrals to quantum operators

        returns:
            Hamiltonian (SparsePauliOp)
        """

        # Get number of orbitals
        norb = self.h1e.shape[0]

        # List of fermionic creation and annihilation operators
        C, D = self.qubit_mapping(2 * norb, mapping="jordan_wigner")

        Exc = []
        for p in range(norb):
            Excp = [C[p] @ D[p] + C[norb + p] @ D[norb + p]]
            for r in range(p + 1, norb):
                Excp.append(
                    C[p] @ D[r]
                    + C[norb + p] @ D[norb + r]
                    + C[r] @ D[p]
                    + C[norb + r] @ D[norb + p]
                )
            Exc.append(Excp)

        # Core term
        H = 0.0 * self.identity(2 * norb)

        # Extra term due to reorganization of two-body Hamiltonian
        t1e = self.h1e - 0.5 * np.einsum("pxrx->pr", self.h2e)

        # One-body term
        for p in range(norb):
            for r in range(p, norb):
                H += t1e[p, r] * Exc[p][r - p]

        # Two-body term
        if (self.use_cholesky):

            # Low-rank decomposition of the Hamiltonian
            Lop, ng = self.cholesky(1e-6)

            for g in range(ng):
                Lg = 0 * self.identity(2 * norb)
                for p in range(norb):
                    for r in range(p, norb):
                        Lg += Lop[p, r, g] * Exc[p][r - p]
                H += 0.5 * Lg @ Lg
        else:
            for p in range(norb):
                for q in range(p, norb):
                    for r in range(norb):
                        for s in range(r, norb):
                            H += 0.5 * self.h2e[p, s, q, r] * Exc[p][q - p] @ Exc[r][s - r]

        return H.chop().simplify()

    def minimize_energy(self, maxiter):

        # Data for adapt-VQE iterations
        iteration = 1
        e_last = self.energy(self.paramstring)
        thr = 1e-6

        print("Initial energy is: %.12f" % e_last)
        a = [0 for q in range(self.nqubits)]
        for iocc in self.ansatz:
            a[iocc] += 1
        ansatz_str = ""
        for q in a:
            ansatz_str += str(q)
        print("Hartree–Fock ansatz is: |%s>" %ansatz_str)
        print("----------------------------------")

        while iteration <= maxiter:

            print("          Iteration %i\n" %iteration)
            
            # List for commutators
            comm_lst = []

            # Compute commutators between the Hamiltonian and the operators in the pool
            for operator in self.operator_pool:
                comm_lst.append(self.commutator(self.hamiltonian, operator))

            # Find operator with largest energy decrease and apply
            self.appended_ops.append(self.operator_pool[comm_lst.index(min(comm_lst))])
            self.paramstring.append(0.0)
            print("Appended operator is:", self.appended_ops[-1].paulis[0])

            # Calculate norm of parameter vector
            print("\nOptimizing parameters ...")
            self.paramstring = list(self.optimize_params())

            # Measure energy
            e = self.energy(self.paramstring)

            print("Energy at iteration %i: %.12f" %(iteration, e))
            e_diff = e - e_last
            print("Ediff = %.12f" %e_diff)

            # Converged?
            if iteration > 1:
                grad_norm = np.sqrt(sum([comm ** 2 for comm in comm_lst]))
                print("Gradient norm = %.10f" %grad_norm)
                #if grad_norm < thr:
                if abs(e_diff) < thr:
                    print("Final energy:", self.energy(self.paramstring))
                    break
            print("\n")

            e_last = e

            iteration += 1
