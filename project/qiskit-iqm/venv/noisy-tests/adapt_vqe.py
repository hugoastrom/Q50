
# Qiskit IQM packages
from iqm.qiskit_iqm import IQMProvider
from iqm.qiskit_iqm.fake_backends import IQMFakeAdonis

# Qiskit packages
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import PauliEvolutionGate, StatePreparation
from qiskit.primitives import Estimator, StatevectorEstimator, BackendEstimatorV2
from qiskit.quantum_info import SparsePauliOp, PauliList
#from qiskit_ibm_runtime import Estimator

# Python standard packages
import numpy as np
import cmath, os
from scipy.optimize import minimize

from quantum_functions import qubit_mapping

class QubitAdaptVQE():
    
    def __init__(self, mol, optimizer, shots = 10000, conv_thr = 1e-6):
        """
        Args:
            mo_occs (list): MO occupations
            h1e (np.ndarray): 1-electron integrals
            h2e (np.ndarray): 2-electron integrals
            optimizer (string): Classical SciPy optimizer algorithm
        """

        self.mol = mol
        
        # Ansatz is an ON vector for 2 * norb spin orbitals
        self.ansatz = self.mol.get_ansatz()
        
        # Declare variables needed
        self.optimizer = optimizer
        self.backend = None
        self.conv_thr = conv_thr
        self.run_on_real_hw = True
        self.shots = shots
        try:
            HELMI_CORTEX_URL = os.getenv('HELMI_CORTEX_URL')
            provider = IQMProvider(HELMI_CORTEX_URL)
            self.backend = provider.get_backend()
        except:
            print("\nNo quantum environment found! Doing classical.")
            self.run_on_real_hw = False

        # Get Hamiltonian
        self.hamiltonian = self.mol.get_hamiltonian()
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

    def classical_to_quantum(self, qc: QuantumCircuit, op: SparsePauliOp):
        """
        Translate SparsePauliOp to measurable quantum circuit
        """

        def apply_basis_rotation(qc, pauli_string):
            #for i, p in enumerate(pauli_string):
            n = len(pauli_string)
            for i in range(n):
                p = pauli_string[n - 1 - i]
                if p == 'X':
                    qc.h(i)
                elif p == 'Y':
                    qc.sdg(i)
                    qc.h(i)
        
        groups = op.group_commuting(qubit_wise=True)

        circuits = []

        #for group in groups:
        for pauli, coeff in zip(op.paulis, op.coeffs):
            qc_copy = qc.copy()

            #rep = group.paulis[0].to_label()
            apply_basis_rotation(qc_copy, pauli.to_label())

            #qc_copy.measure_all()
            creg = ClassicalRegister(self.nqubits)
            qc_copy.add_register(creg)
            qc_copy.measure(range(self.nqubits), range(self.nqubits))
            circuits.append((qc_copy, pauli, coeff))

        return circuits
        
    def run_qc(self, qc: QuantumCircuit, op: SparsePauliOp):

        # If the IQM hardware supports Estimator then do shortcut
        #try:
        #    estimator = Estimator(backend=self.backend)
        #    trans_c = transpile(qc, backend=self.backend)
        #    job = estimator.run(
        #        circuits=[trans_c],
        #        observables=[op],
        #        shots=self.shots
        #    )
        
        #    result = job.result()
        #    return result.values[0]

        #except Exception as e:
        #    print("Estimator failed:", e)

        # Else do it manually
        circuits = self.classical_to_quantum(qc, op)
        total = 0
        for circuit in circuits:
            c = circuit[0]
            trans_c = transpile(c, backend=self.backend, optimization_level=0)
            job = self.backend.run(trans_c, shots=self.shots)
            result = job.result()
            counts = result.get_counts()

            pauli = circuit[1]
            coeff = circuit[2]
            shots = sum(counts.values())
            #for pauli, coeff in zip(paulis, coeffs):
            exp = 0
            label = pauli.to_label()
            n = len(label)
            for bitstring, count in counts.items():
                bitstring = bitstring[::-1]
                parity = 1
                for i in range(n):
                    p = label[n - 1 - i]
                    if p != 'I':
                        if bitstring[i] == '1':
                            parity *= -1
                exp += parity * count / shots
                
            total += coeff.real * exp

        return total


    def calc_exp_val(self, qc: QuantumCircuit, op: SparsePauliOp):
        """
        Calculate expectation value of op
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
        if estimator == "backend_estimator":
            if self.backend == None:
                raise ValueError("Set backend!")
            self.run_on_real_hw = True

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

        values = []
        for _ in range(3):
            values.append(self.calc_exp_val(psi, self.hamiltonian))
            #e = self.calc_exp_val(psi, self.hamiltonian)

        #return e + self.mol.get_hnuc()
        return np.mean(values) + self.mol.get_hnuc()

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

        """ TODO: Implement general operator pools """
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


        # So far only singles and doubles implemented
        norb = self.mol.get_norb()
        nspin = 2 * norb
        C, D = qubit_mapping(nspin)
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
        Initialize quantum circuit for HF ansatz
        """
        qreg = QuantumRegister(self.nqubits)
        psi = QuantumCircuit(qreg)
        for iocc in self.ansatz:
            psi.x(iocc)

        return psi

    def select_operator(self):
        """
        Select operator with largest gradient
        """

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
                if all(two_der >= 0.0 for two_der in second_order):
                    print("Minimum found")
                    break
                else:
                    print("Saddle point, doing second order optimization")

                # Find operator with largest energy decrease and apply
                idx = np.argmax(np.abs(second_order))
                grad = second_order[idx]
            else:
                idx = np.argmax(np.abs(comm_lst))
                grad = comm_lst[idx]

        return self.operator_pool[idx], grad

    
    def minimize_energy(self, maxiter):

        # Check for correct number of electrons
        C, D = qubit_mapping(2 * self.mol.get_norb(), mapping="jordan_wigner")
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
        print("Hartree–Fock ansatz is |%s>" %ansatz_str)
        print("----------------------------------")

        for iteration in range(1, maxiter + 1):

            print("          Iteration %i\n" %iteration)

            operator, grad = self.select_operator()
            self.appended_ops.append(operator)
            self.paramstring.append(0.0)
            print("\nAppended operator is:\n", self.appended_ops[-1], "\nwith gradient:", grad)

            # Reoptimize parameters
            print("\nOptimizing parameters...")
            if iteration == 1:
                print("First iteration: doing grid search")
                thetas = np.linspace(-0.2, 0.2, 21)
                energies = [self.energy([t]) for t in thetas]
                best_theta = thetas[np.argmin(energies)]
                print("   optimal parameter is %.5f" %best_theta)
                self.paramstring.append(best_theta)
            else:
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

        print("\nFinal energy:         %.12f" %self.energy(self.paramstring))
