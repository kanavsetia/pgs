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

from symmetries import find_symmetry_ops
import r_mat_for_mols as r_mats
from int_func import qmol_func

# atom = "O .0 .0 .0; H .757 .586 .0; H -.757 .586 .0"
# atom = "O .0 .0 .0; H 1. .0 .0; H -1. .0 .0"
atom = 'H 0 0 0; H 0 0 .7414'

r_matrices = r_mats.mol_r_matrices('H2', True)

def check_commute(op1, op2):
    op3 = op1 * op2 - op2 * op1
    op3.zeros_coeff_elimination()
    return op3.is_empty()


if __name__ == '__main__':

    mol = gto.Mole()
    mol.atom = atom
    mol.basis = 'sto-3g'
    # mol.basis = {'O': 'sto-3g', 'H': 'cc-pvdz', 'H&#64;2': '6-31G'}

    is_atomic = True
    mol.build()
    _q_ = qmol_func(mol, atomic=is_atomic)
    if is_atomic:
        two_body_temp = QMolecule.twoe_to_spin(_q_.mo_eri_ints)
        mol = gto.M(atom=atom, basis='sto-3g')

        O = get_ovlp(mol)
        X = np.kron(np.identity(2), np.linalg.inv(scipy.linalg.sqrtm(O)))

        fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=two_body_temp)
        fer_op.transform(X)
    else:
        fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=_q_.two_body_integrals)

    ref_op = fer_op.mapping('jordan_wigner')

    ee = ExactEigensolver(ref_op, k=1)
    ee_result = ee.run()
    temp_min_eigvals = ee_result['eigvals']
    print(temp_min_eigvals)
    exit(0)

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
    correct_sector = None
    for taper_coeff in itertools.product([1, -1], repeat=len(single_qubit_list)):
        tapered_qubit_op = Operator.qubit_tapering(v_qubit_op, cliffords, single_qubit_list, list(taper_coeff))
        ee = ExactEigensolver(tapered_qubit_op, k=1)
        ee_result = ee.run()
        temp_min_eigvals = ee_result['eigvals'][0]
        if np.isclose(temp_min_eigvals, ref_min_eigvals, rtol=1e-8):
            correct_sector = list(taper_coeff)
        # print("at sector {}: eig value: {}; reference: {}".format(list(taper_coeff), temp_min_eigvals, ref_min_eigvals.real))

    if correct_sector:
        print("Correct sector is {}.".format(correct_sector))
    else:
        print("None of sectors is correct.")

    ee = ExactEigensolver(v_qubit_op, k=1)
    ee_result = ee.run()
    temp_min_eigvals = ee_result['eigvals'][0]
    is_min_eig_close = np.isclose(ref_min_eigvals, temp_min_eigvals)
    print("after transformed by v matrix, the min eig is {} identical.".format("" if is_min_eig_close else "NOT"))

