from adapt_vqe import QubitAdaptVQE
from adapt_molecule import AdaptMolecule
from iqm.qiskit_iqm.fake_backends import IQMFakeAdonis

import numpy as np
import cmath

from pyscf import ao2mo, gto, mcscf, scf, fci

if __name__ == "__main__":

    # Create PySCF molecule object
    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 2.0;"
    mol.basis = "sto-3g"
    mol.spin = 0
    mol.build()

    # Run PySCF RHF calculation
    mf = scf.RHF(mol).run()
    ehf = mf.e_tot
    e_elec = mf.energy_elec()[0]
    hnuc = ehf - e_elec
    
    # Get one- and two-electron Hamiltonians in MO basis
    c = mf.mo_coeff
    h1 = c.T @ scf.hf.get_hcore(mol) @ c
    eri_ao = mol.intor("int2e")                   # AO integrals
    eri_mo = ao2mo.incore.full(eri_ao, c)         # MO integrals in 8-fold symmetry
    eri_mo = ao2mo.restore(1, eri_mo, c.shape[1]) # Restore full 4-index tensor (ij|kl)
    occs = mf.mo_occ

    # Create qubit-adapt-VQE object
    adapt_mol = AdaptMolecule(occs, hnuc, h1, eri_mo)
    vqe = QubitAdaptVQE(adapt_mol, optimizer="cobyla")

    # Data for qubit-adapt-VQE
    #estimator = "backend_estimator"
    estimator = "statevector_estimator"
    vqe.set_backend(IQMFakeAdonis())
    vqe.set_estimator(estimator)

    # Run qubit-adapt-VQE
    vqe.minimize_energy(maxiter = 50)

    # Print reference FCI energy
    cisolver = fci.FCI(mf)
    efci, fcivec = cisolver.kernel()

    print('FCI reference energy: %.12f' %efci)
    print('HF energy:            %.12f' %ehf)
