
# Qiskit IQM packages
from iqm.qiskit_iqm import IQMProvider
from iqm.qiskit_iqm.fake_backends import IQMFakeAdonis

# Qiskit packages
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate, StatePreparation
from qiskit.primitives import Estimator, StatevectorEstimator, BackendEstimatorV2
from qiskit.quantum_info import SparsePauliOp, PauliList
from qiskit_ibm_runtime import Estimator

# Python packages
import numpy as np
import cmath, os
from scipy.optimize import minimize

class QubitAdaptVQE():
    
    def __init__(self, mo_occs, hnuc, h1e, h2e, optimizer, shots = 1000, cholesky = False, conv_thr = 1e-6, run_on_real_hw = True):
        """
        Args:
            mo_occs (list): MO occupations
            h1e (np.ndarray): 1-electron integrals
            h2e (np.ndarray): 2-electron integrals
            optimizer (string): Classical SciPy optimizer algorithm
        """
        
        # Read input
        norb = h1e.shape[0]
        # Ansatz is an ON vector for 2 * norb spin orbitals
        self.ansatz = []
        for iocc, occ in enumerate(mo_occs):
            if occ > 0:
                self.ansatz.append(iocc)
                if occ > 1:
                    self.ansatz.append(iocc + norb)

        self.hnuc = hnuc
        self.h1e = h1e
        self.h2e = h2e
        self.optimizer = optimizer
        
        # Declare variables needed
        self.backend = None
        self.use_cholesky = cholesky
        self.conv_thr = conv_thr
        self.run_on_real_hw = run_on_real_hw
        self.shots = shots
        if run_on_real_hw:
            try:
                HELMI_CORTEX_URL = os.getenv('HELMI_CORTEX_URL')
                provider = IQMProvider(HELMI_CORTEX_URL)
                self.backend = provider.get_backend()
            except:
                print("No quantum environment found! Doing classical.")
                self.run_on_real_hw = False

        # Build Hamiltonian
        self.hamiltonian = self.build_hamiltonian()
        self.nqubits = self.hamiltonian.num_qubits
        print("Number of qubits = %i\n" %self.nqubits)

        # Construct operator pool and parameters
        self.operator_pool = self.generate_pool(1)
        self.appended_ops = []
        self.paramstring = []

    def estimator_dict(self, estimator):
        d = {"estimator": Estimator(),
             "backend_estimator": BackendEstimatorV2(backend = self.backend),
             "statevector_estimator": StatevectorEstimator()
             }
        return d[estimator]

    def set_backend(self, backend):
        self.backend = backend

    def classical_to_quantum(qc: QuantumCircuit, op: SparsePauliOp):
        """
        Translate SparsePauliOp to measurable quantum circuit
        """

        def apply_basis_rotation(qc, pauli_string):
            for i, p in enumerate(pauli_string):
                if p == 'X':
                    qc.h(i)
                elif p == 'Y':
                    qc.sdg(i)
                    qc.h(i)
        
        groups = op.group_commuting()

        circuits = []

        for group in groups:
            qc_copy = qc.copy()

            for pauli in group.paulis:
                apply_basis_rotation(qc_copy, pauli.to_label())

            qc_copy.measure_all()
            circuits.append((qc_copy, group))

        return circuits
        
    def run_qc(self, qc: QuantumCircuit, op: SparsePauliOp):

        # If the IQM hardware supports Estimator then do shortcut
        try:
            estimator = Estimator(backend=self.backend)
            trans_c = transpile(qc, backend=self.backend)
            job = estimator.run(
                circuits=[trans_c],
                observables=[op],
                shots=self.shots
            )
        
            result = job.result()
            return result.values[0]

        except Exception as e:
            print("Estimator failed:", e)

        # Else do it manually
        circuits = self.classical_to_quantum(qc, op)
        total = 0
        for circuit in circuits:
            c = circuit[0]
            trans_c = transpile(c, backend=self.backend)
            job = self.backend.run(trans_c, shots=self.shots)
            result = job.result()
            counts = result.get_counts()

            pauli_group = circuit[1]
            for pauli, coeff in zip(pauli_group.paulis, pauli_group.coeffs):
                exp = 0
                shots = sum(counts.values())
                for bitstring, count in counts.items():
                    parity = 1
                    for i, p in enumerate(pauli):
                        if p != 'I':
                            bit = bitstring[-1 - i]
                            if bit == '1':
                                parity *= -1
                    exp += parity * count / shots

                total += coeff * exp

        return total


    def calc_exp_val(self, qc: QuantumCircuit, op: SparsePauliOp):
        """
        Calculate expectation value of op classically using qc
        args:
            qc (QuntumCircuit): The state
            op (SparsePauliOp): The observable
        """
        if self.run_on_real_hw:
            res = self.run_qc(qc, op)
        else:
            estimator = self.estimator
            if isinstance(estimator, Estimator):
                res = estimator.run([qc], [op]).result().values
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
        for op, theta in zip(self.appended_ops, self.paramstring):
            evolution = PauliEvolutionGate(op, time=theta)
            psi.compose(evolution, inplace=True)

        operator = a @ b - b @ a
        # The derivative of the energy with respect to parameters yields a complex phase
        operator = -1.0j * operator.chop().simplify()
        # Measure value
        exp_val = self.calc_exp_val(psi, operator)

        return exp_val

    def energy(self, params):
        """
        Measure energy
        """
        psi = self.state_prep()
        # Need to declare the parameters explicitly for SciPy optimization routine
        for op, theta in zip(self.appended_ops, params):
            evolution = PauliEvolutionGate(op, time=theta)
            psi.compose(evolution, inplace=True)

        e = self.calc_exp_val(psi, self.hamiltonian)

        return e + self.hnuc

    def optimize_params(self, args={"disp": True}):
        """
        Classically optimize parameter vector
        args:
            args (dict): Optional arguments for the SciPy optimizer
        returns:
            Optimized parameter vector
        """

        res = minimize(self.energy, self.paramstring, method = self.optimizer, tol=1e-6, options=args)

        return res.x

    def generate_pool(self, pool_type):
        """
        Generate operator pool
        """

        #if pool_type == 0:
        #    op_strings = ["ZY", "YI"]
        #    for q in range(2, self.nqubits):
        #        op_strings = ["Z" + op for op in op_strings] + ["Y" + "I" * q, "IY" + "I" * (q - 1)]
        #elif pool_type == 1:
        #    op_strings = []
        #    for p in range(self.nqubits - 1):
        #        q = self.nqubits - p - 2
        #        op_strings.append("I" * p + "ZY" + "I" * q)
        #    for p in range(self.nqubits - 1):
        #        q = self.nqubits - p - 1
        #        op_strings.append("I" * p + "Y" + "I" * q)
        #else:
        #    raise ValueError("Pool type not implemented")

        #op_strings = ["XYXY"], "YYXX", "XYYX", "YXXY"]
        #pool = [SparsePauliOp(op) for op in op_strings]
        #norb = self.h1e.shape[0]
        # List of fermionic creation and annihilation operators
        #C, D = self.qubit_mapping(2 * norb, mapping="jordan_wigner")
        #pool = [ C[2] @ C[3] @ D[0] @ D[1] - C[1] @ C[0] @ D[3] @ D[2] ]
        #pool = [1.0j * ((C[1] @ D[0] + C[3] @ D[2]) - (C[0] @ D[1] + C[2] @ D[3])).chop().simplify(), 1.0j * (C[2] @ C[3] @ D[0] @ D[1] - C[1] @ C[0] @ D[3] @ D[2]).chop().simplify()]
        #print(pool)

        norb = self.h1e.shape[0]
        nspin = 2 * norb
        C, D = self.qubit_mapping(nspin)
        
        pool = []
        
        # Singles
        for i in range(nspin):
            for a in range(nspin):
                if i >= a:
                    continue
                op = 1j * (C[a] @ D[i] - C[i] @ D[a])
                pool.append(op.chop().simplify())

        # Doubles
        for i in range(nspin):
            for j in range(i+1, nspin):
                for a in range(nspin):
                    for b in range(a+1, nspin):
                        op = 1j * (C[a] @ C[b] @ D[i] @ D[j] - C[j] @ C[i] @ D[b] @ D[a])
                        pool.append(op.chop().simplify())

        return pool

    def state_prep(self):
        """
        Initialize quantum circuit
        """
        psi = QuantumCircuit(self.nqubits)
        for iocc in self.ansatz:
            psi.x(iocc)

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
            from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
            from qiskit_nature.second_q.mappers import JordanWignerMapper
            
            # build Hamiltonian
            hamiltonian = ElectronicEnergy.from_raw_integrals(self.h1e, self.h2e)
            
            # fermionic operator
            fermionic_op = hamiltonian.second_q_op()
            
            # map to qubits
            mapper = JordanWignerMapper()
            qubit_op = mapper.map(fermionic_op)
            H = qubit_op

        return H.chop().simplify()

    def minimize_energy(self, maxiter):

        # Check for correct number of electrons
        norb = self.h1e.shape[0]
        C, D = self.qubit_mapping(2 * norb, mapping="jordan_wigner")
        N_op = sum(C[p] @ D[p] for p in range(self.nqubits))
        print("Number of electrons =", self.calc_exp_val(self.state_prep(), N_op))
        
        # Data for adapt-VQE iterations
        e_last = self.energy(self.paramstring)

        print("Initial energy is: %.12f" % e_last)
        a = [0 for q in range(self.nqubits)]
        for iocc in self.ansatz:
            a[iocc] += 1
        ansatz_str = ""
        for q in a:
            ansatz_str += str(q)
        print("Hartree–Fock ansatz |\alpha_1\alpha_2...\alpha_n\beta_1\beta_2...\beta_m> = |%s>" %ansatz_str)
        print("----------------------------------")

        for iteration in range(1, maxiter + 1):

            print("          Iteration %i\n" %iteration)
            
            # List for commutators
            comm_lst = []

            # Compute commutators between the Hamiltonian and the operators in the pool
            for operator in self.operator_pool:
                comm_val = self.commutator(self.hamiltonian, operator)
                comm_lst.append(np.real_if_close(comm_val))

            # Check for saddle point
            if all(abs(comm) < 1e-8 for comm in comm_lst):
                print("\nAll gradients are zero, doing second derivatives")
                second_order = []
                for operator in self.operator_pool:
                    second_order.append(float(self.commutator(-1.0j * operator, self.hamiltonian @ operator - operator @ self.hamiltonian)))
                print(second_order)
                if all(two_der >= 0.0 for two_der in second_order):
                    print("Minimum found")
                    break
                else:
                    print("Saddle point, doing second order optimization")

                # Find operator with largest energy decrease and apply
                idx = np.argmax(np.abs(second_order))
                grad = second_order[idx]
                self.appended_ops.append(self.operator_pool[idx])
            else:
                idx = np.argmax(np.abs(comm_lst))
                grad = comm_lst[idx]
                self.appended_ops.append(self.operator_pool[idx])
            self.paramstring.append(0.0)
            print("\nAppended operator is:\n", self.appended_ops[-1], "\nwith gradient:", grad)

            # Calculate norm of parameter vector
            print("\nOptimizing parameters...")
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
                if grad_norm < self.conv_thr:
                    break
            print("\n")

            e_last = e

            iteration += 1
        print("\nFinal energy:         %.12f" %self.energy(self.paramstring))
