
import itertools

import numpy as np
from scipy import linalg as scila
# from qiskit.aqua import Operator
import qutip as qt
from qiskit.quantum_info import Pauli
import pdb

def find_symmetry_ops(r_matrices):
    """

    Args:
        r_matrices (list[numpy.ndarray]): a list of rotation matrices.

    Returns:
        numpy.ndarray: the V matrix
        list[Operator]: symmetry paulis
        list[Operator]: cliffords, composed of symmetries and single-qubit op
        list[int]: position of the single-qubit operators that anticommute
            with the cliffords

    """
    modes = r_matrices[0].shape[0]

    g_matrices = []
    for r_matrix in r_matrices:
        g_matrix = -1j * scila.logm(r_matrix)
        g_matrices.append(g_matrix)

    sim_dia = []
    for g_matrix in g_matrices:
        sim_dia.append(qt.Qobj(g_matrix))

    d_v = qt.simdiag(sim_dia)

    d_matrices = d_v[0]
    v_matrix = np.hstack([d_v[1][i].data.toarray() for i in range(modes)])

    # check the build d_matrix
    for eig in d_matrices.flatten():
        print(eig)
        if not (np.isclose(eig, 0.0) or np.isclose(eig, np.pi)):
            # print(np.where(d_matrices.flatten()==eig))
            raise ValueError('The specified R matrix is invalid. \
                                        Eigenvalues of G includes: {}'.format(eig))
    single_qubit_list = []
    cliffords = []
    pauli_symmetries = []
    existed_pi_locs = []
    # pdb.set_trace()
    for d_idx in range(len(d_matrices)):
        pi_index = np.where(np.isclose(d_matrices[d_idx], np.pi))[0]
        single_qubit_pauli = ['I'] * modes
        pi_loc = 0
        for i in pi_index:
            if i not in existed_pi_locs:
                pi_loc = i
                existed_pi_locs.extend(pi_index.tolist())
                break
        single_qubit_pauli[pi_loc] = 'X'
        single_qubit_pauli = ''.join(single_qubit_pauli)
        single_qubit_op = Operator(paulis=[[1.0, Pauli.from_label(single_qubit_pauli[::-1])]])
        single_qubit_list.append(pi_loc)
        symmetries_pauli_label = ''
        for i in range(modes):
            symmetries_pauli_label += 'I' if i not in pi_index else 'Z'
        sym_pauli = Pauli.from_label(symmetries_pauli_label[::-1])
        pauli_symmetries.append(sym_pauli)
        symmetries_op = Operator(paulis=[[1.0, sym_pauli]])
        clifford_op = single_qubit_op + symmetries_op
        clifford_op.scaling_coeff(1.0 / np.sqrt(2))
        cliffords.append(clifford_op)
    
    return v_matrix, pauli_symmetries, cliffords, single_qubit_list
