from adapt_vqe import QubitAdaptVQE
from iqm.qiskit_iqm.fake_backends import IQMFakeAdonis

import numpy as np
import cmath

from pyscf import ao2mo, gto, mcscf, scf, lo
from pyscf.tools import dump_mat

if __name__ == "__main__":

    # Create molecule object
    mol = gto.Mole()
    mol.atom = "He 0 0 0"
    mol.basis = "sto-3g"
    mol.spin = 0
    mol.build()

    # Run RHF calculation
    mf = scf.RHF(mol).run()
    print("RHF energy = %.12f" %mf.e_tot)

    # One- and two-electron Hamiltonians in MO basis
    h1 = mf.mo_coeff.T @ scf.hf.get_hcore(mol) @ mf.mo_coeff
    eri_ao = mol.intor("int2e")                     # AO integrals
    eri_mo = ao2mo.incore.full(eri_ao, mf.mo_coeff) # MO integrals in 8-fold symmetry
    eri_mo = ao2mo.restore(1, eri_mo, mol.nao)      # Restore full 4-index tensor (ij|kl)

    # Create qubit-adapt-VQE object
    occs = mf.mo_occ

    vqe = QubitAdaptVQE(occs, h1, eri_mo, optimizer="cobyla")

    # Data for qubit-adapt-VQE
    estimator = "statevector_estimator"
    vqe.set_backend(IQMFakeAdonis())
    vqe.set_estimator(estimator)

    # Run qubit-adapt-VQE
    vqe.minimize_energy(maxiter = 100)
