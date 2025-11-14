from adapt_vqe import QubitAdaptVQE
from iqm.qiskit_iqm.fake_backends import IQMFakeAdonis

import numpy as np
import cmath

from pyscf import ao2mo, gto, mcscf, scf, lo
from pyscf.tools import dump_mat

if __name__ == "__main__":

    # Create molecule object
    #a = 1.5
    #mol = gto.Mole()
    #mol.build(
    #    verbose = 0,
    #    atom = [["H", (0, 0, 0)], ["H", (0, 0, a)]],
    #    basis = "sto-3g",
    #    spin = 0,
    #    charge = 0,
        #symmetry = "Dooh"
    #)

    # Run classical SCF to obtain ansatz
    #mf = scf.RHF(mol)
    #mf.scf()
    #e_hf = mf.energy_tot()
    #print("HF energy:", e_hf)

    # Orbitals
    #dump_mat.dump_mo(mol, mf.mo_coeff)
    
    # Define active space
    #active_space = range(mol.nelectron // 2 - 1, mol.nelectron // 2 + 1)
    #active_space = range(4)
    #active_space = range(10)


    # Get 1 and 2 electron Hamiltonians
    #hcore = mf.get_hcore()
    #ecore = 0.0
    #h2e = mol.intor("int2e")#, aosym="s8")
    #E1 = mf.kernel()
    #mx = mcscf.CASCI(mf, ncas=len(active_space), nelecas=(1, 1))
    #mo = mx.sort_mo(active_space, base=0)
    #E2 = mx.kernel(mo)[:2]
    #h1e, ecore = mx.get_h1eff()
    #h2e = ao2mo.restore(1, mx.get_h2eff(), mx.ncas)

    mol = gto.M(atom='He 0 0 0', basis='sto-3g')
    conv, e, mo_e, mo, mo_occ = scf.hf.kernel(scf.hf.SCF(mol), dm0=np.eye(mol.nao_nr()))
    uhf_mol = scf.UHF(mol)
    #uhf_mol.kernel()
    print('conv = %s, E(HF) = %.12f' % (conv, e))

    dm1 = scf.hf.make_rdm1(mo, mo_occ)
    dm2 = scf.hf.make_rdm2(mo, mo_occ)
    hcore = scf.hf.get_hcore(mol)
    #dm1 = sum(uhf_mol.make_rdm1())
    #dm2 = (np.einsum('ij,kl->ijkl', dm1, dm1) - np.einsum('ij,kl->iklj', dm1, dm1)/2)
    #dm2 = sum(uhf_mol.make_rdm2())
    #hcore = uhf_mol.get_hcore()
    eri = mol.intor("int2e")
    E = np.einsum('pq,qp', hcore, dm1) + np.einsum('pqrs,pqrs', eri, dm2) / 2
    print("             E(HF) = %.12f" % (E))

    for estimator in ["estimator"]:#, "statevector_estimator", "backend_estimator"):
        # Create qubit-adapt-VQE object
        vqe = QubitAdaptVQE(dm1, hcore, eri, optimizer="cobyla")

        # Data for qubit-adapt-VQE
        vqe.set_backend(IQMFakeAdonis())
        vqe.set_estimator(estimator)

        # Prepare initial state and run
        print("Initial energy:", vqe.energy([0.0]))
        
        # adapt-VQE iterations
        vqe.minimize_energy(maxiter = 100)
