import logging
import copy
import itertools
from qiskit.quantum_info import Pauli
from pyscf import gto, scf, ao2mo
from pyscf.lib import param
from scipy import linalg as scila
from pyscf.lib import logger as pylogger
from qiskit.chemistry import QMolecule
import numpy as np
from qiskit.aqua import Operator
from qiskit.aqua.algorithms import ExactEigensolver
import qutip as qt
from qiskit.chemistry import FermionicOperator
logger = logging.getLogger(__name__)
import int_func
from r_mat_for_mols import mol_r_matrices, check_commute
np.set_printoptions(linewidth=230,suppress=True,precision=3,threshold=5000)

class r_mat_funcs():

	def __init__(self, MoleculeFlag,check_r_matrix_flag):

		self.MoleculeFlag = MoleculeFlag
		[self.r_matrices,self.fer_op]=mol_r_matrices(MoleculeFlag,check_r_matrix_flag)

	

	def sim_diag(self, r_matrices):
		r_qt_ob = []
		for r_matrix in r_matrices:
			r_qt_ob.append(qt.Qobj(r_matrix))

		r_mat_diag = qt.simdiag(r_qt_ob)
		r_mat_evals = r_mat_diag[0]
		# r_mat_evals = r_mat_evals[np.argsort(np.sum(np.where(np.round(r_mat_evals,10)==-1.0,r_mat_evals,np.zeros(np.shape(r_mat_evals))),axis=1))[::-1],:]
		v_matrix=np.hstack([r_mat_diag[1][i].data.toarray() for i in range(self.fer_op.modes)])
		# print(v_matrix)
		print("checking the v matrix...")
		is_identity = np.allclose(np.dot(v_matrix, v_matrix.conj().T), np.identity(self.fer_op.modes))
		print("v matrix is {} unitary.".format("" if is_identity else "NOT"))
		return [r_mat_evals,v_matrix]

	def sym_transf_ham_qub_op(self, v_matrix):
		temp_fer_op = copy.deepcopy(self.fer_op)
		temp_fer_op.transform(v_matrix)
		v_qubit_op = temp_fer_op.mapping(map_type='jordan_wigner')
		return v_qubit_op
	
	def ind_symm_r_ev_mat(self,r_mat_evals):
		r_F2 = np.zeros(np.shape(r_mat_evals))
		for i in range(np.shape(r_mat_evals)[0]):
			for j in range(np.shape(r_mat_evals)[1]):
				if np.round(r_mat_evals[i,j],10)==-1.:
					# print(i,j)
					r_F2[i,j]=1

		r_F21 = np.zeros([np.shape(r_F2)[0]+1,np.shape(r_F2)[1]])
		r_F21[0:np.shape(r_F2)[0],:] = r_F2
		r_F21[np.shape(r_F2)[0],:]=np.ones(np.shape(r_F2)[1])

		r_mat_evals = Operator.row_echelon_F2(r_F21)

		r_mat_ev_z = np.sum(r_mat_evals,1)
		r_mat_evals = np.delete(r_mat_evals, np.where(r_mat_ev_z==0),axis=0)
		print(r_mat_evals)
		return r_mat_evals

	def check_commute(self, op1, op2):
		op3 = op1 * op2 - op2 * op1
		op3.zeros_coeff_elimination()
		return op3.is_empty()

	def get_symm_list(self, r_mat_evals):
		sym_list = []
		for i in range(np.shape(r_mat_evals)[0]):
			sym_str = ''
			for j in range(np.shape(r_mat_evals)[1]):
				if np.round(r_mat_evals[i,j],10)==1.:
					sym_str+='Z'
				else:
					sym_str+='I'
			sym_pauli = Pauli.from_label(sym_str[::-1])
			sym_list.append(sym_pauli)
		return sym_list
	
	def get_cliffords(self, r_mat_evals, sym_list):
		X_list = []
		X_op_lis = []
		uni_transf = []
		Z_ind_list = []
		
		for i1 in range(np.shape(r_mat_evals)[0]):
			Z_ind = np.where(np.round(r_mat_evals[i1,:],10)==1)
			Z_ind_list.append(Z_ind)
		

		for i in range(np.shape(r_mat_evals)[0]):
			Z_ind = np.where(np.round(r_mat_evals[i,:],10)==1)
			# X_array=np.array(X_list)
			# print(Z_ind)
			for j in range(np.shape(Z_ind[0])[0]):
				try :
					# X_list.index(Z_ind[0][j])
					if np.shape(r_mat_evals)[0]==1:
						raise ValueError

					for t in range(np.shape(r_mat_evals)[0]):
						v = True
						if t != i:
							pass
							print(bool(np.where(Z_ind_list[t]==Z_ind[0][j])[0].any()))
							v = v and not(bool(np.where(Z_ind_list[t]==Z_ind[0][j])[0].any()))
						
						if v==True:
							raise ValueError
							
				except ValueError:
					X_list.append(Z_ind[0][j])
					# print(Z_ind[0][j])
					X_str = 'I'*int(Z_ind[0][j])+'X'+'I'*int((self.fer_op.modes-1-Z_ind[0][j]))
					X_pauli = Pauli.from_label(X_str[::-1])
					X_op = Operator(paulis=[[1.0, X_pauli]])
					X_op_lis.append(X_op)
					Z_op = Operator(paulis=[[1.0, sym_list[i]]])
					U_op = X_op +  Z_op
					U_op.scaling_coeff(1.0 / np.sqrt(2))
					# print(U_op.print_operators())
					uni_transf.append(U_op)
					# print(X_str)
					should_be_i = U_op*U_op
					break
		return [uni_transf, X_list]


	def get_tapered_qubit_op(self, v_qubit_op, uni_transf, X_list, tapper_coeffs):
		tapered_qubit_op = Operator.qubit_tapering(v_qubit_op, uni_transf, X_list, tapper_coeffs)
		return tapered_qubit_op


def qubit_tapering(operator, cliffords, sq_list, tapering_values):
        if len(cliffords) == 0 or len(sq_list) == 0 or len(tapering_values) == 0:
            logger.warning("Cliffords, single qubit list and tapering values cannot be empty.\n"
                           "Return the original operator instead.")
            return operator

        if len(cliffords) != len(sq_list):
            logger.warning("Number of Clifford unitaries has to be the same as length of single"
                           "qubit list and tapering values.\n"
                           "Return the original operator instead.")
            return operator
        if len(sq_list) != len(tapering_values):
            logger.warning("Number of Clifford unitaries has to be the same as length of single"
                           "qubit list and tapering values.\n"
                           "Return the original operator instead.")
            return operator

        if operator.is_empty():
            logger.warning("The operator is empty, return the empty operator directly.")
            return operator

        operator.to_paulis()

        for clifford in cliffords:
            operator = clifford * operator * clifford
        operator.zeros_coeff_elimination()
        operator_out = Operator(paulis=[])
        X_1 = range(14)
        def diff(first, second):
            second = set(second)
            return [item for item in first if item not in second]
        
        # print(diff(X_1,X_list))
        
        for pauli_term in operator.paulis:
            coeff_out = pauli_term[0]
            for idx, qubit_idx in enumerate(sq_list):
                if not (not pauli_term[1].z[qubit_idx] and not pauli_term[1].x[qubit_idx]):
                    coeff_out = tapering_values[idx] * coeff_out
            z_temp = np.delete(pauli_term[1].z.copy(), np.asarray(sq_list))
            x_temp = np.delete(pauli_term[1].x.copy(), np.asarray(sq_list))
            # print(sq_list)
            z_temp_1 = pauli_term[1].z[sq_list]
            x_temp_1 = pauli_term[1].x[sq_list]
            p = Pauli(z_temp_1, x_temp_1)
            # print(p.to_label())
            # print(coeff_out)
            pauli_term_out = [coeff_out, Pauli(z_temp, x_temp)]
            operator_out += Operator(paulis=[pauli_term_out])

        operator_out.zeros_coeff_elimination()
        return operator_out

	
x = r_mat_funcs('BF3', True)
[r_mat_evals,v_matrix] = x.sim_diag(x.r_matrices)
v_qubit_op = x.sym_transf_ham_qub_op(v_matrix)
r_mat_evals = x.ind_symm_r_ev_mat(r_mat_evals)
sym_list = x.get_symm_list(r_mat_evals)
print("check the commutativity of the found symmetry paulis between H'.")
for symm in sym_list:
	symm_op = Operator(paulis=[[1.0, symm]])
	is_commuted = check_commute(symm_op, v_qubit_op)
	print(symm_op.print_operators())
	print("symm is {} commuted.".format("" if is_commuted else "NOT"))

[uni_transf,X_list] = x.get_cliffords(r_mat_evals,sym_list)
print(X_list)
tapper_coeffs = np.ones([len(sym_list),1])
tapered_qubit_op = x.get_tapered_qubit_op(v_qubit_op,uni_transf,X_list,tapper_coeffs)

