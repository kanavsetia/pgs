import copy

import numpy as np
from qiskit.aqua import Operator
from qiskit.chemistry import FermionicOperator, QMolecule
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.aqua.algorithms import ExactEigensolver

from pyscf import gto
from pyscf.scf.hf import get_ovlp
import scipy

from symmetries import find_symmetry_ops

r_matrices = []

# R-matrix for plane of symmetry \sigma_{xy}. Everything remains the same, only pz-orbitals pick up negative sign.
r = np.eye(14)
r[4, 4] = -1
r[11, 11] = -1
r_matrices.append(r)
# R-matrix for plane of symmetry \sigma_{xz}. Everything remains the same, only py-orbitals pick up negative sign.
r = np.eye(14)
r[3, 3] = -1
r[10, 10] = -1
r_matrices.append(r)

# R-matrix for plane of symmetry \sigma_{yz}. Everything remains the same, only px-orbitals pick up negative sign and hydrogen atoms swap.
r=np.zeros([14,14])
r[0,0]=1
r[1,1]=1
r[2,2]=-1
r[3,3]=1
r[4,4]=1
r[5,6]=1
r[6,5]=1
r[7,7]=1
r[8,8]=1
r[9,9]=-1
r[10,10]=1
r[11,11]=1
r[12,13]=1
r[13,12]=1
r_matrices.append(r)

# R-matrix for symmetry-axis C_2. Linear water molecule has three axis of symmetry:
# About z-axis
r = np.zeros((14,14))
r[0,0]=1
r[1,1]=1
r[2,2]=-1
r[3,3]=-1
r[4,4]=1
r[5,6]=1
r[6,5]=1
r[7,7]=1
r[8,8]=1
r[9,9]=-1
r[10,10]=-1
r[11,11]=1
r[12,13]=1
r[13,12]=1
r_matrices.append(r)
# About y-axis
r = np.zeros((14,14))
r[0,0]=1
r[1,1]=1
r[2,2]=-1
r[3,3]=1
r[4,4]=-1
r[5,6]=1
r[6,5]=1
r[7,7]=1
r[8,8]=1
r[9,9]=-1
r[10,10]=1
r[11,11]=-1
r[12,13]=1
r[13,12]=1
r_matrices.append(r)
#Symmetry about x-axis:
r = np.zeros((14,14))
r[0,0]=1
r[1,1]=1
r[2,2]=1
r[3,3]=-1
r[4,4]=-1
r[5,6]=1
r[6,5]=1
r[7,7]=1
r[8,8]=1
r[9,9]=1
r[10,10]=-1
r[11,11]=-1
r[12,13]=1
r[13,12]=1
r_matrices.append(r)

#Spin symmetry:
r = np.zeros((14, 14))
for i in range(14):
    if i < 7:
        r[i+7,i]= 1.
    else:
        r[i-7,i] = 1.
r_matrices.append(r)

def check_commute(op1, op2):
    op3 = op1 * op2 - op2 * op1
    op3.zeros_coeff_elimination()
    return op3.is_empty()


if __name__ == '__main__':
    is_atomic = True
    basis = 'sto3g'
    mol_string = "O .0 .0 .0; H 1. .0 .0; H -1. .0 .0"
    pyscf_driver = PySCFDriver(mol_string, unit=UnitsType.ANGSTROM, charge=0,
                               spin=0, basis=basis, is_atomic=is_atomic)

    molecule = pyscf_driver.run()

    if is_atomic:
        temp_int = np.einsum('ijkl->ljik', molecule.mo_eri_ints)
        two_body_temp = QMolecule.twoe_to_spin(temp_int)
        mol = gto.M(atom=mol_string, basis=basis)

        O = get_ovlp(mol)
        X = np.kron(np.identity(2), np.linalg.inv(scipy.linalg.sqrtm(O)))

        fer_op = FermionicOperator(h1=molecule.one_body_integrals, h2=two_body_temp)
        fer_op.transform(X)
    else:
        fer_op = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)

    print("checking r matrices...")
    for r_matrix in r_matrices:
        temp_fer_op = copy.deepcopy(fer_op)
        temp_fer_op.transform(r_matrix)

        if np.all(np.isclose(np.abs(temp_fer_op.h1 - fer_op.h1), 0.0)) and \
                np.all(np.isclose(np.abs(temp_fer_op.h2 - fer_op.h2), 0.0)):
            print("r matrix is okay.")
        else:
            print("This r matrix is wrong r matrix:\n{}".format(r_matrix))

    ref_op = fer_op.mapping('jordan_wigner')

    v_matrix, pauli_symmetries, cliffords, single_qubit_list = find_symmetry_ops(r_matrices)

    ee = ExactEigensolver(ref_op, k=1)
    ee_result = ee.run()
    ref_min_eigvals = ee_result['eigvals'][0]

    print("checking the v matrix...")
    is_identity = np.allclose(np.dot(v_matrix, v_matrix.T), np.identity(fer_op.modes))
    print("v matrix is {} identity.".format("" if is_identity else "NOT"))
    temp_fer_op = copy.deepcopy(fer_op)
    temp_fer_op.transform(v_matrix)
    v_qubit_op = temp_fer_op.mapping(map_type='jordan_wigner')
    # v_qubit_op.to_paulis()
    print("check the commutativity of the found symmetry paulis.")
    for symm in pauli_symmetries:
        symm_op = Operator(paulis=[[1.0, symm]])
        is_commuted = check_commute(symm_op, v_qubit_op)
        print("symm is {} commuted.".format("" if is_commuted else "NOT"))

    ee = ExactEigensolver(v_qubit_op, k=1)
    ee_result = ee.run()
    temp_min_eigvals = ee_result['eigvals'][0]
    is_min_eig_close = np.isclose(ref_min_eigvals, temp_min_eigvals)
    print("after transformed by v matrix, the min eig is {} identical.".format("" if is_min_eig_close else "NOT"))
