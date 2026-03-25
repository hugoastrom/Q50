
# Python standard packages
import numpy as np

# Qiskit
from qiskit.quantum_info import SparsePauliOp

class AdaptMolecule():

    def __init__(self, mo_occs, hnuc, h1e, h2e, cholesky = False):
        """
        Args:
            mo_occs (list): MO occupations
            h1e (np.ndarray): 1-electron integrals
            h2e (np.ndarray): 2-electron integrals
        """

        """ Read input """
        self.norb = h1e.shape[0]
        # Ansatz is an ON vector for 2 * norb spin orbitals
        self.ansatz = []
        for iocc, occ in enumerate(mo_occs):
            if occ > 0:
                self.ansatz.append(iocc)
                if occ > 1:
                    self.ansatz.append(iocc + self.norb)
        self.hnuc = hnuc
        self.h1e = h1e
        self.h2e = h2e

        
        # Build Hamiltonian
        self.use_cholesky = cholesky
        self.hamiltonian = self.build_hamiltonian()


    def get_hamiltonian(self):
        return self.hamiltonian

    def get_norb(self):
        return self.norb

    def get_hnuc(self):
        return self.hnuc

    def get_ansatz(self):
        return self.ansatz
    
    def build_hamiltonian(self) -> SparsePauliOp:
        """
        Map one- and two-electron integrals to quantum operators

        returns:
            Hamiltonian (SparsePauliOp)
        """

        # List of fermionic creation and annihilation operators
        #C, D = self.qubit_mapping(2 * self.norb, mapping="jordan_wigner")

        #Exc = []
        #for p in range(self.norb):
        #    Excp = [C[p] @ D[p] + C[self.norb + p] @ D[self.norb + p]]
        #    for r in range(p + 1, self.norb):
        #        Excp.append(
        #            C[p] @ D[r]
        #            + C[self.norb + p] @ D[self.norb + r]
        #            + C[r] @ D[p]
        #            + C[self.norb + r] @ D[self.norb + p]
        #        )
        #    Exc.append(Excp)

        # Core term
        #H = 0.0 * self.identity(2 * self.norb)
        # Extra term due to reorganization of two-body Hamiltonian
        #t1e = self.h1e - 0.5 * np.einsum("pxrx->pr", self.h2e)

        # One-body term
        #for p in range(self.norb):
        #    for r in range(p, self.norb):
        #        H += t1e[p, r] * Exc[p][r - p]

        # Two-body term
        if (self.use_cholesky):

            # Low-rank decomposition of the Hamiltonian
            Lop, ng = self.cholesky(1e-6)

            for g in range(ng):
                Lg = 0 * self.identity(2 * self.norb)
                for p in range(self.norb):
                    for r in range(p, self.norb):
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
