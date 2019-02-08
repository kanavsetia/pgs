import copy
import itertools

import numpy as np
from qiskit.aqua import Operator
from qiskit.chemistry import FermionicOperator, QMolecule
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.aqua.algorithms import ExactEigensolver

from pyscf import gto
from pyscf.scf.hf import get_ovlp
import scipy

from qiskit.chemistry.mole_geo_symmetry.symmetries import find_symmetry_ops

from qiskit.chemistry.mole_geo_symmetry.int_func import qmol_func

try_h2o = False
try_l_h2o = True
try_h2 = False
r_matrices = []

if try_h2o:
    # atom = "O .0 .0 .0; H .757 .586 .0; H -.757 .586 .0"
    atom = [['O', (0.0, 0.0, 0.0)], ['H', (.757, .586, 0)], ['H', (-.757, .586, 0.0)]]

    r = np.zeros((14, 14))
    # Spin symmetry:
    for i in range(14):
       if i<7:
           r[i+7,i]=1.
       else:
           r[i-7,i]=1.
    r_matrices.append(r)

    # R-matrix for plane of symmetry \sigma_{xy}. Everything remains the same, only pz-orbitals pick up negative sign.
    r = np.eye(14)
    r[4, 4] = -1
    r[11, 11] = -1
    r_matrices.append(r)

    # R-matrix for plane of symmetry \sigma_{yz}. Everything remains the same, only px-orbitals pick up negative sign and the hydrogen atoms swap.
    r = np.eye(14)
    r[2, 2] = -1
    r[9, 9] = -1
    r[12, 12] = 0
    r[13, 13] = 0
    r[12, 13] = 1
    r[13, 12] = 1
    r[5, 6] = 1
    r[6, 5] = 1
    r[5, 5] = 0
    r[6, 6] = 0
    r_matrices.append(r)

    # Axial symmetry about y-axis
    r=np.zeros([14,14])
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

if try_l_h2o:
    # atom = "O .0 .0 .0; H 1. .0 .0; H -1. .0 .0"
    atom = [['O',(0.0, 0.0,0.0)],['H',(1, 0, 0)], ['H',(-1.0,0.0,0.0)]]
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

    # # R-matrix for symmetry-axis C_2. Linear water molecule has three axis of symmetry:
    # # About z-axis
    # r = np.zeros((14,14))
    # r[0,0]=1
    # r[1,1]=1
    # r[2,2]=-1
    # r[3,3]=-1
    # r[4,4]=1
    # r[5,6]=1
    # r[6,5]=1
    # r[7,7]=1
    # r[8,8]=1
    # r[9,9]=-1
    # r[10,10]=-1
    # r[11,11]=1
    # r[12,13]=1
    # r[13,12]=1
    # r_matrices.append(r)
    # # About y-axis
    # r = np.zeros((14,14))
    # r[0,0]=1
    # r[1,1]=1
    # r[2,2]=-1
    # r[3,3]=1
    # r[4,4]=-1
    # r[5,6]=1
    # r[6,5]=1
    # r[7,7]=1
    # r[8,8]=1
    # r[9,9]=-1
    # r[10,10]=1
    # r[11,11]=-1
    # r[12,13]=1
    # r[13,12]=1
    # r_matrices.append(r)
    # #Symmetry about x-axis:
    # r = np.zeros((14,14))
    # r[0, 0] = 1
    # r[1, 1] = 1
    # r[2, 2] = 1
    # r[3, 3] = -1
    # r[4, 4] = -1
    # r[5, 5] = 1
    # r[6, 6] = 1
    # r[7, 7] = 1
    # r[8, 8] = 1
    # r[9, 9] = 1
    # r[10, 10] = -1
    # r[11, 11] = -1
    # r[12, 12] = 1
    # r[13, 13] = 1
    # r_matrices.append(r)

    #Spin symmetry:
    r = np.zeros((14, 14))
    for i in range(14):
        if i < 7:
            r[i+7,i]= 1.
        else:
            r[i-7,i] = 1.
    r_matrices.append(r)

if try_h2:
    # atom = 'H 0 0 0; H 0 0 .7414'
    atom = [['H', (.0, .0, .0)], ['H', (.0, .0, 0.7414)]]
    # Defining R-matrix --> r
    # Swapping the spatial orbitals. This involves swapping both the spin orbitals corresponding to a spatial orbital.
    r = np.zeros((4, 4))
    r[0,1]=1
    r[1,0]=1
    r[2,3]=1
    r[3,2]=1
    r_matrices.append(r)

    # Swapping the spin oritals. Spin symmetry.
    r=np.zeros((4,4))
    for i in range(4):
        if i < 2:
            r[i + 2, i] = 1.
        else:
            r[i - 2, i] = 1.
    r_matrices.append(r)

def check_commute(op1, op2):
    op3 = op1 * op2 - op2 * op1
    op3.zeros_coeff_elimination()
    return op3.is_empty()


if __name__ == '__main__':

    mol = gto.Mole()
    mol.atom = atom
    mol.basis = 'sto-3g'
    # mol.basis = {'O': 'sto-3g', 'H': 'cc-pvdz', 'H&#64;2': '6-31G'}

    mol.build()
    _q_ = qmol_func(mol, atomic=True)
    fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=_q_.two_body_integrals)

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
    print("single_qubit_list: {}".format(single_qubit_list))

    print("checking the v matrix...")
    is_identity = np.allclose(np.dot(v_matrix, v_matrix.T), np.identity(fer_op.modes))
    print("v matrix is {} unitary.".format("" if is_identity else "NOT"))
    temp_fer_op = copy.deepcopy(fer_op)
    temp_fer_op.transform(v_matrix)
    v_qubit_op = temp_fer_op.mapping(map_type='jordan_wigner')
    print("check the commutativity of the found symmetry paulis between H'.")
    for symm in pauli_symmetries:
        symm_op = Operator(paulis=[[1.0, symm]])
        is_commuted = check_commute(symm_op, v_qubit_op)
        print(symm_op.print_operators())
        print("symm is {} commuted.".format("" if is_commuted else "NOT"))

    print("check the commutativity of found symmetry paulis each other.")
    for i in range(len(pauli_symmetries)):
        for j in range(i):
            symm_op_i = Operator(paulis=[[1.0, pauli_symmetries[i]]])
            symm_op_j = Operator(paulis=[[1.0, pauli_symmetries[j]]])
            is_commuted = check_commute(symm_op_i, symm_op_j)
            print("symm ({}, {}) is {} commuted.".format(i, j, "" if is_commuted else "NOT"))

    ee = ExactEigensolver(ref_op, k=1)
    ee_result = ee.run()
    ref_min_eigvals = ee_result['eigvals'][0]

    print("Trying to tapering")

    for taper_coeff in itertools.product([1, -1], repeat=len(single_qubit_list) - 1):
        tapered_qubit_op = Operator.qubit_tapering(v_qubit_op, cliffords[:-1], single_qubit_list[:-1], list(taper_coeff))
        ee = ExactEigensolver(tapered_qubit_op, k=1)
        ee_result = ee.run()
        temp_min_eigvals = ee_result['eigvals'][0].real
        print("at sector {}: eig value: {}; reference: {}".format(list(taper_coeff), temp_min_eigvals, ref_min_eigvals.real))

    ee = ExactEigensolver(v_qubit_op, k=1)
    ee_result = ee.run()
    temp_min_eigvals = ee_result['eigvals'][0]
    is_min_eig_close = np.isclose(ref_min_eigvals, temp_min_eigvals)
    print(temp_min_eigvals.real)
    print("after transformed by v matrix, the min eig is {} identical.".format("" if is_min_eig_close else "NOT"))

