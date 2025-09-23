
import numpy as np

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats import MoleculeInfo
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.transformers import FreezeCoreTransformer

from qiskit_algorithms.optimizers import SLSQP
from qiskit_algorithms.minimum_eigensolvers import AdaptVQE, VQE

from qiskit.primitives import Estimator

# Use PySCF, a classical computational chemistry software
# package, to compute the one-body and two-body integrals in
# molecular-orbital basis, necessary to form the Fermionic operator

info = MoleculeInfo(["Li", "H"], [(0.0, 0.0, 0.0), (0.0, 0.0, 1.5)])

driver = PySCFDriver.from_molecule(info, basis="sto3g")
molecule = driver.run()

transformer = FreezeCoreTransformer()
molecule = transformer.transform(molecule)
hamiltonian = molecule.hamiltonian.second_q_op()
mapper = ParityMapper(num_particles=molecule.num_particles)
tapered_mapper = molecule.get_tapered_mapper(mapper)
qubit_op = tapered_mapper.map(hamiltonian)



# setup the initial state for the variational form
init_state = HartreeFock(
            molecule.num_spatial_orbitals,
            molecule.num_particles,
            tapered_mapper,
        )

estimator = Estimator()

optimizer = SLSQP(maxiter=10000, ftol=1e-9)



vqe_ansatz = UCCSD(
    molecule.num_spatial_orbitals,
    molecule.num_particles,
    tapered_mapper,
    initial_state=init_state
)
vqe = VQE(estimator, vqe_ansatz, optimizer)
vqe.initial_point = [0] * vqe_ansatz.num_parameters
algo = GroundStateEigensolver(tapered_mapper, vqe)
result_vqe = algo.solve(molecule)
energy_vqe = result_vqe.eigenvalues[0]



adapt_vqe_ansatz = UCCSD(
    molecule.num_spatial_orbitals,
    molecule.num_particles,
    tapered_mapper,
    initial_state=init_state
)

adapt_vqe = AdaptVQE(VQE(estimator, adapt_vqe_ansatz, optimizer))
result_adapt_vqe = adapt_vqe.compute_minimum_eigenvalue(qubit_op)
energy_adapt_vqe = result_adapt_vqe.eigenvalue

from prettytable import PrettyTable

table = PrettyTable()
table.field_names = ["Ansatz","Energy (Hartree)","Gates"]

vqe_cirq = result_vqe.raw_result.optimal_circuit
table.add_row(['UCCSD', str(energy_vqe), vqe_cirq.count_ops()])

adapt_vqe_cirq = result_adapt_vqe.optimal_circuit
table.add_row(['ADAPT-VQE', str(result_adapt_vqe.eigenvalue), adapt_vqe_cirq.count_ops()])

print(table)
