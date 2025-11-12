from adapt_vqe import QubitAdaptVQE
from iqm.qiskit_iqm.fake_backends import IQMFakeAdonis

import numpy as np
import cmath

from pyscf import ao2mo, gto, mcscf, scf, lo
from pyscf.tools import dump_mat

if __name__ == "__main__":

    # Create molecule object
    a = 1.5
    mol = gto.Mole()
    mol.build(
        verbose = 0,
        atom = [["Li", (0, 0, 0)], ["H", (0, 0, a)]],
        basis = "sto-3g",
        spin = 0,
        charge = 0,
        #symmetry = "Dooh"
    )

    # Run classical SCF to obtain ansatz
    mf = scf.RHF(mol)
    mf.scf()
    e_hf = mf.energy_tot()
    print("HF energy:", e_hf)

    # Orbitals
    #dump_mat.dump_mo(mol, mf.mo_coeff)
    
    # Define active space
    active_space = range(mol.nelectron // 2 - 1, mol.nelectron // 2 + 1)
    #active_space = range(4)
    #active_space = range(10)


    # Get 1 and 2 electron Hamiltonians
    #ecore = mf.get_hcore()
    #h1e = mf.get_ovlp()
    #h2e = mol.intor("int2e", aosym="s8")
    E1 = mf.kernel()
    mx = mcscf.CASCI(mf, ncas=len(active_space), nelecas=(1, 1))
    mo = mx.sort_mo(active_space, base=0)
    E2 = mx.kernel(mo)[:2]
    h1e, ecore = mx.get_h1eff()
    h2e = ao2mo.restore(1, mx.get_h2eff(), mx.ncas)
    #h2e = mx.get_h2eff()

    for estimator in ("estimator", "statevector_estimator", "backend_estimator"):
        # Create qubit-adapt-VQE object
        vqe = QubitAdaptVQE(ecore, h1e, h2e, optimizer="cobyla")

        # Data for qubit-adapt-VQE
        vqe.set_backend(IQMFakeAdonis())
        vqe.set_estimator(estimator)

        # Prepare initial state and run
        print("Initial energy:", vqe.energy([0.0]))
        
        # adapt-VQE iterations
        vqe.minimize_energy(maxiter = 100)
