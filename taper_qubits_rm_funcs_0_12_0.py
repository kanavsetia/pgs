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
from qiskit.aqua.operators import WeightedPauliOperator as WPO
from qiskit.aqua.operators import Z2Symmetries, WeightedPauliOperator
from qiskit.aqua.operators import common
from qiskit.aqua.algorithms import ExactEigensolver
import qutip as qt
from qiskit.chemistry import FermionicOperator
logger = logging.getLogger(__name__)
import int_func
import time
from r_mat_for_mols import mol_r_matrices, check_commute
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.algorithms.adaptive import VQE
from qiskit import BasicAer
from qiskit.aqua import set_qiskit_aqua_logging, QuantumInstance
np.set_printoptions(linewidth=230,suppress=True,precision=3,threshold=5000)
from qiskit.aqua import set_qiskit_aqua_logging
import logging
set_qiskit_aqua_logging(logging.INFO)

class r_mat_funcs():

	def __init__(self, MoleculeFlag,check_r_matrix_flag,is_atomic):

		self.MoleculeFlag = MoleculeFlag
		[self.r_matrices,self.fer_op, self.num_particles]=mol_r_matrices(MoleculeFlag,check_r_matrix_flag,is_atomic)

	
	def sim_diag(self, r_matrices):
		r_qt_ob = []
		for r_matrix in r_matrices:
			r_qt_ob.append(qt.Qobj(r_matrix))

		r_mat_diag = qt.simdiag(r_qt_ob)
		# r_mat_diag = qt.simdiag(r_matrices)
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
					r_F2[i,j]=1

		r_F21 = np.zeros([np.shape(r_F2)[0]+1,np.shape(r_F2)[1]])
		r_F21[0:np.shape(r_F2)[0],:] = r_F2
		r_F21[np.shape(r_F2)[0],:]=np.ones(np.shape(r_F2)[1])
		
		r_mat_evals = common.row_echelon_F2(r_F21)
		r_mat_ev_z = np.sum(r_mat_evals,1)
		r_mat_evals = np.delete(r_mat_evals, np.where(r_mat_ev_z==0),axis=0)
		return r_mat_evals

	def check_commute(self, op1, op2):
		op3 = op1 * op2 - op2 * op1
		op3.chop()
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
		print(r_mat_evals)
		for sym in sym_list:
			print(sym.to_label())
		X_list = []
		X_op_lis = []
		X_p_lis = []
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
							# print(bool(np.where(Z_ind_list[t]==Z_ind[0][j])[0].any()))
							v = v and not(bool(np.where(Z_ind_list[t]==Z_ind[0][j])[0].any()))
						
						if v==True:
							raise ValueError
							
				except ValueError:
					X_list.append(Z_ind[0][j])
					X_str = 'I'*int(Z_ind[0][j])+'X'+'I'*int((self.fer_op.modes-1-Z_ind[0][j]))
					X_pauli = Pauli.from_label(X_str[::-1])
					X_p_lis.append(X_pauli)
					X_op = WPO(paulis=[[1.0, X_pauli]])
					X_op_lis.append(X_op)
					Z_op = WPO(paulis=[[1.0, sym_list[i]]])
					U_op = X_op +  Z_op
					U_op._scaling_weight(1.0 / np.sqrt(2))
					# print(U_op.print_operators())
					uni_transf.append(U_op)
					# print(X_str)
					should_be_i = U_op*U_op
					break
		return [uni_transf, X_list, X_p_lis]

	def get_tapered_qubit_op(self, v_qubit_op, uni_transf, X_list, tapper_coeffs):
		tapered_qubit_op = WPO.qubit_tapering(v_qubit_op, uni_transf, X_list, tapper_coeffs)
		return tapered_qubit_op


if __name__ == "__main__":

	AO = True
	MF = 'BeH2'

	run_vqe = False
	check_ref_energy = True
	if AO == True:
		x = r_mat_funcs(MF, True,True)
        # Simultaneously diagonalize the r-matrices.
		[r_mat_evals,v_matrix] = x.sim_diag(x.r_matrices)
		# This renumbering of r_matrices and v_matrix is just for the specific case of BeH2
		if MF == 'BeH2':
			V_tr = np.zeros([14,14])
			V_tr[0,0]=V_tr[1,1]=V_tr[2,6]=V_tr[3,10]=V_tr[4,12]=V_tr[5,2]=V_tr[6,7]= 1
			V_tr[7,4]=V_tr[8,5]=V_tr[9,9]=V_tr[10,11]=V_tr[11,13]=V_tr[12,3]=V_tr[13,8]= 1
			r1=np.zeros([14,14])
			r1[0:4,:]=r_mat_evals
			v_matrix = np.matmul(v_matrix,V_tr.T)
			# print(r1)
			r1 = np.dot(r1,V_tr.T)
			r_mat_evals = r1[0:4,:]
		#################################################
        # transform the Hamiltonian using the eigenvectors of the r-matrices
        # Getting and transforming the Ham with V matrix.
		v_qubit_op = x.sym_transf_ham_qub_op(v_matrix)
		qub_op = x.fer_op.mapping('jordan_wigner')
        # Find the symmetries using the find_z2_symm method. These will be the same as 
        # the ones found using the symmeries.
        # [symmetries, sq_paulis, cliffords, sq_list] = qub_op.find_Z2_symmetries()
		# print('Z2 symmetries found:')
		# for symm in symmetries:
		# 	print(symm.to_label())
		# 	sym_la = symm.to_label()[::-1]
		# 	ind = [i for i, a in enumerate(sym_la) if a == 'Z']
		# 	print(ind)
		# print('single qubit operators found:')
		# exit()

		# qub_op.chop()
		print('Number of terms in the Hamiltonian in AO basis')
		print(len(qub_op._paulis))
		# ref_min_eigvals = -1
		#Checkin to make sure everything works
		if check_ref_energy ==True:
			ee = ExactEigensolver(qub_op, k=6)
			ee_result = ee.run()
			ref_min_eigvals = ee_result['eigvals'][0:6]
			# This is the reference value from Hamiltonian in AO basis
			print('Eigenvalue of the full Ham in AO basis')
			print(ref_min_eigvals)
			exit()
			# input()

		#Printing the number of terms in the Hamiltonian
		print('Final number of terms in the Hamiltonian in AO basis after the transformation')
		print(len(v_qubit_op._paulis))
		# The set of symmetries is not independent, so, the following code gets the independent set of symmetries.
		ind_symm_mat = x.ind_symm_r_ev_mat(r_mat_evals)
		sym_list = x.get_symm_list(ind_symm_mat)

		# In order to check if the symmetries commute with Ham, uncomment the following piece of code:
		print("check the commutativity of the found symmetry paulis between H'.")
		for symm in sym_list:
			symm_op = WPO(paulis=[[1.0, symm]])
			is_commuted = check_commute(symm_op, v_qubit_op)
			sym_la = symm.to_label()[::-1]
            # qubit numbers for which the symmetru acts non trivially. Could be printed for a sanity check
			ind = [i for i, a in enumerate(sym_la) if a == 'Z']
			print("symm is {} commuted.".format("" if is_commuted else "NOT"))

		# Get the unitary operators (cliffords) corresponding the single qubit string.

		[cliffords, single_qubit_list, X_op_lis] = x.get_cliffords(ind_symm_mat,sym_list)
		print('Following are the qubits which are tappered off.')
		print(single_qubit_list)

		print("Trying to tapering")
		z2symm = Z2Symmetries(sym_list, X_op_lis, single_qubit_list)
		the_tapered_ops = z2symm.taper(v_qubit_op)

		i=0
		for coeff in itertools.product([1, -1], repeat=len(z2symm._sq_list)):
			op = the_tapered_ops[i]
			ee = ExactEigensolver(op, k=1)
			ee_result = ee.run()
			temp_min_eigvals = ee_result['eigvals'][0]
			if np.isclose(temp_min_eigvals.real, ref_min_eigvals[0].real, rtol=1e-8):
				correct_sector = list(coeff)
				correct_op = op
		#     print("eig value: {}".format(temp_min_eigvals))
			print("at sector {}: eig value: {}; reference: {}".format(list(coeff), temp_min_eigvals, ref_min_eigvals[0].real))
			i=i+1

		# correct_sector=[1.,1.,1.,-1.,-1.]

		# Get the tappered qubit operator
		# tapered_qubit_op = x.get_tapered_qubit_op(v_qubit_op,cliffords,single_qubit_list,correct_sector)
		tapered_qubit_op = correct_op
		ee = ExactEigensolver(tapered_qubit_op.copy(), k=6)
		ee_result = ee.run()
		print('Getting the eigen values of the tappered off qubit operator')
		print(ee_result['eigvals'][0:6])
		exit()
		if run_vqe==True:
			init_state = HartreeFock(num_qubits=qub_op.num_qubits- len(single_qubit_list), num_orbitals=x.fer_op.modes,
								qubit_mapping='jordan_wigner', two_qubit_reduction=False,
								num_particles=x.num_particles, sq_list=single_qubit_list)

			# setup variationl form
			var_form = UCCSD(num_qubits=qub_op.num_qubits- len(single_qubit_list), depth=1,
							num_orbitals=x.fer_op.modes, 
							num_particles=x.num_particles,
							active_occupied=None, active_unoccupied=None, initial_state=init_state,
							qubit_mapping='jordan_wigner', two_qubit_reduction=False, 
							num_time_slices=1,
							#    )
							cliffords=cliffords, sq_list=single_qubit_list, tapering_values=correct_sector, symmetries=sym_list)

			# var_form = RYRZ(num_qubits=qub_op.num_qubits- len(single_qubit_list), depth=30,entanglement='linear')
			# print(type(var_form))
			# setup optimizer
			optimizer = COBYLA(maxiter=2000)
			# import pdb; pdb.set_trace()
			# set vqe
			algo = VQE(tapered_qubit_op, var_form, optimizer, 'matrix')

			# setup backend
			backend = BasicAer.get_backend('statevector_simulator')
			quantum_instance = QuantumInstance(backend=backend)


			algo_result = algo.run(quantum_instance)


			print(algo_result['energy'])


	################## MO basis##############################################
	else:
		x = r_mat_funcs(MF, False,False)
		qub_op = x.fer_op.mapping('jordan_wigner')
		print(len(qub_op._paulis))
		# exit()
		if check_ref_energy:
		#Checkin to make sure everything works
			ee = ExactEigensolver(qub_op.copy(),k=1)
			ee_result = ee.run()
			ref_min_eigvals = ee_result['eigvals'][0]
			# This is the reference value from Hamiltonian in MO basis
			print('Eigenvalue of the full Ham in MO basis')
			print(ref_min_eigvals)
		# exit()
		[symmetries, sq_paulis, cliffords, sq_list] = qub_op.find_Z2_symmetries()

		print('Z2 symmetries found:')
		for symm in symmetries:
			print(symm.to_label())
			sym_la = symm.to_label()[::-1]
			ind = [i for i, a in enumerate(sym_la) if a == 'Z']
			print(ind)
		print('single qubit operators found:')
		for sq in sq_paulis:
			print(sq.to_label())
		print('cliffords found:')
		for clifford in cliffords:
			print(clifford.print_operators())
		print('single-qubit list: {}'.format(sq_list))


		tapered_qubit_op = Operator.qubit_tapering(qub_op, cliffords, sq_list, np.ones(len(sq_list)))
		print(len(tapered_qubit_op._paulis))

		print("Trying to taper")
		# correct_sector = None
		# for taper_coeff in itertools.product([1, -1], repeat=len(sq_list)):
		# 	tapered_qubit_op = Operator.qubit_tapering(qub_op, cliffords, sq_list, list(taper_coeff))
		# 	ee = ExactEigensolver(tapered_qubit_op, k=1)
		# 	ee_result = ee.run()
		# 	temp_min_eigvals = ee_result['eigvals'][0]
		# 	if np.isclose(temp_min_eigvals, ref_min_eigvals, rtol=1e-8):
		# 		correct_sector = list(taper_coeff)
		# 	print("at sector {}: eig value: {}; reference: {}".format(list(taper_coeff), temp_min_eigvals, ref_min_eigvals.real))

		correct_sector=[-1, 1, 1, 1, -1]

		# Get the tappered qubit operator
		tapered_qubit_op = x.get_tapered_qubit_op(qub_op,cliffords,sq_list,correct_sector)
		# ee = ExactEigensolver(tapered_qubit_op.copy(), k=6)
		# ee_result = ee.run()
		# print('Getting the eigen values of the tappered off qubit operator')
		# print(ee_result['eigvals'][0:6])

		if run_vqe:
			init_state = HartreeFock(num_qubits=qub_op.num_qubits- len(sq_list), num_orbitals=x.fer_op.modes,
								qubit_mapping='jordan_wigner', two_qubit_reduction=False,
								num_particles=x.num_particles, sq_list=sq_list)

			# setup variationl form
			var_form = UCCSD(num_qubits=qub_op.num_qubits- len(sq_list), depth=1,
							num_orbitals=x.fer_op.modes, 
							num_particles=x.num_particles,
							active_occupied=None, active_unoccupied=None, initial_state=init_state,
							qubit_mapping='jordan_wigner', two_qubit_reduction=False, 
							num_time_slices=1,
							#    )
							cliffords=cliffords, sq_list=sq_list, tapering_values=correct_sector, symmetries=symmetries)

			# print(type(var_form))
			# setup optimizer
			optimizer = COBYLA(maxiter=1500)
			# import pdb; pdb.set_trace()
			# set vqe
			algo = VQE(tapered_qubit_op, var_form, optimizer, 'matrix')

			# setup backend
			backend = BasicAer.get_backend('statevector_simulator')
			quantum_instance = QuantumInstance(backend=backend)

			algo_result = algo.run(quantum_instance)

			print(algo_result['energy'])

