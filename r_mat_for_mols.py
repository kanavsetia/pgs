# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 14:46:51 2018

@author: kanav
"""
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
# from qiskit.aqua import Operator
from qiskit.aqua.operators import WeightedPauliOperator as WPO

from qiskit.aqua.algorithms import ExactEigensolver
import scipy
from pyscf.scf.hf import get_ovlp
from symmetries import find_symmetry_ops
from qiskit.chemistry import FermionicOperator
from int_func import qmol_func
logger = logging.getLogger(__name__)
import int_func

def mol_r_matrices(MoleculeFlag,check_r_matrix_flag,is_atomic):

	mol = gto.Mole()
	#=================================
	# Hydrogen molecule
	#=================================

	if MoleculeFlag == 'H2':
		r_matrices=[]
		num_particles = 2
		if is_atomic:
			try:
				data = np.load(MoleculeFlag+'ao.npz')
				data.files
				one_b = data['one_b']
				two_b = data['two_b']
			except IOError:
				mol.atom = [['H',(0, 0, -0.3707)], ['H',(0,0.0,0.3707)]]
				mol.build()
				_q_=int_func.qmol_func(mol, atomic=True)
				one_b=_q_.one_body_integrals
				two_b=_q_.two_body_integrals
				# np.savez(MoleculeFlag+'_ao.npz',one_b=_q_.one_body_integrals,two_b=_q_.two_body_integrals)
		
			two_body_temp = QMolecule.twoe_to_spin(_q_.mo_eri_ints)
			temp_int = np.einsum('ijkl->ljik', _q_.mo_eri_ints)
			two_body_temp = QMolecule.twoe_to_spin(temp_int)
			mol = gto.M(atom=mol.atom, basis='sto-3g')

			O = get_ovlp(mol)
			X = np.kron(np.identity(2), np.linalg.inv(scipy.linalg.sqrtm(O)))

			fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=two_body_temp)
			fer_op.transform(X)
		else:
			mol.atom = [['H',(0, 0, -0.3707)], ['H',(0,0.0,0.3707)]]
			mol.build()
			_q_=int_func.qmol_func(mol, atomic=is_atomic)
			one_b=_q_.one_body_integrals
			two_b=_q_.two_body_integrals
			fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=_q_.two_body_integrals)


		one_b = fer_op.h1
		two_b = fer_op.h2


		# Defining R-matrix --> r
		# Swapping the spatial orbitals. This involves swapping both the spin orbitals corresponding to a spatial orbital.
		# This could be treated as a reflection symmetry or rotational symmetry.
		r1 = np.zeros([4,4])
		r1[0,1]=1
		r1[1,0]=1
		r1[2,3]=1
		r1[3,2]=1

		# Swapping the spin oritals. Spin symmetry.
		r2=np.zeros([4,4])
		for i in range(4):
			if i<2:
				r2[i+2,i]=1.
			else:
				r2[i-2,i]=1.
		
		r_matrices.append(r1)
		r_matrices.append(r2)
		if check_r_matrix_flag:
			if bool(check_r_mat(r_matrices,fer_op,one_b,two_b)):
				print('All the above matrices work!')


	#=================================
	# Water molecule (with different basis sets)
	#=================================

	elif MoleculeFlag== 'H2O_L':
		r_matrices = []
		print(MoleculeFlag)
		# Configuration from 
		# mol.atom = [['O',(0.8638, 0.4573,0.0)], ['H',(0, 0, 0)], ['H',(1.7785,0.0,0.0)]]
		# mol.atom = [['O',(0.0, 0.0,0.0)],['H',(1, 0, 0)], ['H',(-1.0,0.0,0.0)]]
		#mol.atom = [['O',(0, 0, 0)], ['H',(0, 1, 0)], ['H@2',(0, 0, 1)]]
		#mol.basis = {'O': 'sto-3g', 'H': 'cc-pvdz', 'H@2': '6-31G'}
		
		num_particles=10
		
		if is_atomic:
			try:
				data = np.load(MoleculeFlag+'ao.npz')
				data.files
				one_b = data['one_b']
				two_b = data['two_b']
			except IOError:
				mol.atom = [['O',(0.0, 0.0,0.0)],['H',(1, 0, 0)], ['H',(-1.0,0.0,0.0)]]
				mol.build()
				_q_=int_func.qmol_func(mol, atomic=True)
				one_b=_q_.one_body_integrals
				two_b=_q_.two_body_integrals
				# np.savez(MoleculeFlag+'_ao.npz',one_b=_q_.one_body_integrals,two_b=_q_.two_body_integrals)
		
			two_body_temp = QMolecule.twoe_to_spin(_q_.mo_eri_ints)
			temp_int = np.einsum('ijkl->ljik', _q_.mo_eri_ints)
			two_body_temp = QMolecule.twoe_to_spin(temp_int)
			mol = gto.M(atom=mol.atom, basis='sto-3g')

			O = get_ovlp(mol)
			X = np.kron(np.identity(2), np.linalg.inv(scipy.linalg.sqrtm(O)))

			fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=two_body_temp)
			fer_op.transform(X)
		else:
			mol.atom = [['O',(0.0, 0.0,0.0)],['H',(1, 0, 0)], ['H',(-1.0,0.0,0.0)]]
			mol.build()
			_q_=int_func.qmol_func(mol, atomic=is_atomic)
			one_b=_q_.one_body_integrals
			two_b=_q_.two_body_integrals
			fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=_q_.two_body_integrals)


		one_b = fer_op.h1
		two_b = fer_op.h2

		 
		# R-matrix for plane of symmetry \sigma_{xy}. Everything remains the same, only pz-orbitals pick up negative sign.
		r1=np.zeros([14,14])
		r1[0,0]=1
		r1[1,1]=1
		r1[2,2]=1
		r1[3,3]=1
		r1[4,4]=-1
		r1[5,5]=1
		r1[6,6]=1
		r1[7,7]=1
		r1[8,8]=1
		r1[9,9]=1
		r1[10,10]=1
		r1[11,11]=-1
		r1[12,12]=1
		r1[13,13]=1

		r_matrices.append(r1)
		# R-matrix for plane of symmetry \sigma_{xz}. Everything remains the same, only py-orbitals pick up negative sign.
		r2=np.eye(14)
		r2[3,3]=-1
		r2[10,10]=-1
		r_matrices.append(r2)

		# R-matrix for plane of symmetry \sigma_{yz}. Everything remains the same, only px-orbitals pick up negative sign and hydrogen atoms swap.
		r3=np.zeros([14,14])
		r3[0,0]=1
		r3[1,1]=1
		r3[2,2]=-1
		r3[3,3]=1
		r3[4,4]=1
		r3[5,6]=1
		r3[6,5]=1
		r3[7,7]=1
		r3[8,8]=1
		r3[9,9]=-1
		r3[10,10]=1
		r3[11,11]=1
		r3[12,13]=1
		r3[13,12]=1
		r_matrices.append(r3)

		# R-matrix for symmetry-axis C_2. Linear water molecule has three axis of symmetry:
		# About z-axis
		r4=np.zeros([14,14])
		r4[0,0]=1
		r4[1,1]=1
		r4[2,2]=-1
		r4[3,3]=-1
		r4[4,4]=1
		r4[5,6]=1
		r4[6,5]=1
		r4[7,7]=1
		r4[8,8]=1
		r4[9,9]=-1
		r4[10,10]=-1
		r4[11,11]=1
		r4[12,13]=1
		r4[13,12]=1
		r_matrices.append(r4)
		#About y-axis
		r5=np.zeros([14,14])
		r5[0,0]=1
		r5[1,1]=1
		r5[2,2]=-1
		r5[3,3]=1
		r5[4,4]=-1
		r5[5,6]=1
		r5[6,5]=1
		r5[7,7]=1
		r5[8,8]=1
		r5[9,9]=-1
		r5[10,10]=1
		r5[11,11]=-1
		r5[12,13]=1
		r5[13,12]=1
		r_matrices.append(r5)
		#Symmetry about x-axis:
		r6=np.zeros([14,14])
		r6[0,0]=1
		r6[1,1]=1
		r6[2,2]=1
		r6[3,3]=-1
		r6[4,4]=-1
		r6[5,5]=1
		r6[6,6]=1
		r6[7,7]=1
		r6[8,8]=1
		r6[9,9]=1
		r6[10,10]=-1
		r6[11,11]=-1
		r6[12,12]=1
		r6[13,13]=1
		r_matrices.append(r6)
		
		#Spin symmetry:
		r7=np.zeros([14,14])
		for i in range(14):
			if i<7:
				r7[i+7,i]=1.
			else:
				r7[i-7,i]=1.
		r_matrices.append(r7)
		if check_r_matrix_flag:
			if bool(check_r_mat(r_matrices,fer_op,one_b,two_b)):
				print('All the above matrices work!')

	#=================================
	# Water molecule (with different basis sets)
	#=================================

	elif MoleculeFlag== 'H2O':
		r_matrices=[]
		print(MoleculeFlag)

		num_particles = 10

		if is_atomic:
			try:
				data = np.load(MoleculeFlag+'ao.npz')
				data.files
				one_b = data['one_b']
				two_b = data['two_b']
			except IOError:
				mol.atom = [['O',(0.0000, 0.0000, 0.0000)],
							['H',(0.757, 0.586, 0.0)],
							['H',(-0.757, 0.586, 0.0)]]

				# mol.atom = [['N', (0.0000,  0.0000, 0.0000)],   
					# ['H', (0.0000,	-1.,-0.3816)],  
					# ['H', (0.8,	0.6	,-0.3816)],  
					# ['H', (-0.8,	0.6	,-0.3816)]]		
				mol.build()
				_q_=int_func.qmol_func(mol, atomic=True)
				one_b=_q_.one_body_integrals
				two_b=_q_.two_body_integrals
				# np.savez(MoleculeFlag+'_ao.npz',one_b=_q_.one_body_integrals,two_b=_q_.two_body_integrals)
		
			two_body_temp = QMolecule.twoe_to_spin(_q_.mo_eri_ints)
			temp_int = np.einsum('ijkl->ljik', _q_.mo_eri_ints)
			two_body_temp = QMolecule.twoe_to_spin(temp_int)
			mol = gto.M(atom=mol.atom, basis='sto-3g')

			O = get_ovlp(mol)
			X = np.kron(np.identity(2), np.linalg.inv(scipy.linalg.sqrtm(O)))

			fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=two_body_temp)
			fer_op.transform(X)
		else:
			mol.atom = [['O',(0.0000, 0.0000, 0.0000)],
					['H',(0.757, 0.586, 0.0)],
					['H',(-0.757, 0.586, 0.0)]]
			mol.build()
			_q_=int_func.qmol_func(mol, atomic=is_atomic)
			one_b=_q_.one_body_integrals
			two_b=_q_.two_body_integrals
			fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=_q_.two_body_integrals)


		one_b = fer_op.h1
		two_b = fer_op.h2

		

		#Spin symmetry:
		r1=np.zeros([14,14])
		for i in range(14):
			if i<7:
				r1[i+7,i]=1.
			else:
				r1[i-7,i]=1.
		# r_matrices.append(r1)
		# R-matrix for plane of symmetry \sigma_{xy}. Everything remains the same, only pz-orbitals pick up negative sign.
		r2=np.eye(14)
		r2[4,4]=-1
		r2[11,11]=-1
		r_matrices.append(r2)
		# R-matrix for plane of symmetry \sigma_{yz}. Everything remains the same, only px-orbitals pick up negative sign and the hydrogen atoms swap.
		r3=np.eye(14)
		r3[2,2]=-1
		r3[9,9]=-1
		r3[12,12]=0
		r3[13,13]=0
		r3[12,13]=1
		r3[13,12]=1
		r3[5,6]=1
		r3[6,5]=1
		r3[5,5]=0
		r3[6,6]=0
		# print(r)
		r_matrices.append(r3)
		#Axial symmetry about y-axis
		r4=np.zeros([14,14])
		r4[0,0]=1
		r4[1,1]=1
		r4[2,2]=-1
		r4[3,3]=1
		r4[4,4]=-1
		r4[5,6]=1
		r4[6,5]=1
		r4[7,7]=1
		r4[8,8]=1
		r4[9,9]=-1
		r4[10,10]=1
		r4[11,11]=-1
		r4[12,13]=1
		r4[13,12]=1
		r_matrices.append(r4)
		if check_r_matrix_flag:
			if bool(check_r_mat(r_matrices,fer_op,one_b,two_b)):
				print('All the above matrices work!')

	#=================================
		# Ammonia molecule
	#=================================

	elif MoleculeFlag=='NH3':
		print(MoleculeFlag)
		num_particles = 10

		if is_atomic:
			try:
				data = np.load(MoleculeFlag+'ao.npz')
				data.files
				one_b = data['one_b']
				two_b = data['two_b']
			except IOError:
				mol.atom = [['N' ,  ( 0.0000000,    0.0000000,    0.1493220)],
							['H' ,  ( 0.0000000 ,   0.9474830 ,   -0.3484190)],
							['H' ,  ( 0.8205440  ,  -0.4737420 ,   -0.3484190)],
							['H' ,  ( -0.8205440  ,  -0.4737420 ,   -0.3484190)]]
				mol.build()
				_q_=int_func.qmol_func(mol, atomic=True)
				one_b=_q_.one_body_integrals
				two_b=_q_.two_body_integrals
				# np.savez(MoleculeFlag+'_ao.npz',one_b=_q_.one_body_integrals,two_b=_q_.two_body_integrals)
		
			two_body_temp = QMolecule.twoe_to_spin(_q_.mo_eri_ints)
			temp_int = np.einsum('ijkl->ljik', _q_.mo_eri_ints)
			two_body_temp = QMolecule.twoe_to_spin(temp_int)
			mol = gto.M(atom=mol.atom, basis='sto-3g')

			O = get_ovlp(mol)
			X = np.kron(np.identity(2), np.linalg.inv(scipy.linalg.sqrtm(O)))

			fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=two_body_temp)
			fer_op.transform(X)
		else:
			mol.atom = [['N' ,  ( 0.0000000,    0.0000000,    0.1493220)],
						['H' ,  ( 0.0000000 ,   0.9474830 ,   -0.3484190)],
						['H' ,  ( 0.8205440  ,  -0.4737420 ,   -0.3484190)],
						['H' ,  ( -0.8205440  ,  -0.4737420 ,   -0.3484190)]]
			mol.build()
			_q_=int_func.qmol_func(mol, atomic=is_atomic)
			one_b=_q_.one_body_integrals
			two_b=_q_.two_body_integrals
			fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=_q_.two_body_integrals)


		one_b = fer_op.h1
		two_b = fer_op.h2
		r_matrices=[]
		#Spin symmetry:
		r1=np.zeros([16,16])
		for i in range(16):
			if i<8:
				r1[i+8,i]=1.
			else:
				r1[i-8,i]=1.
		r_matrices.append(r1)
		# R-matrix for plane of symmetry \sigma_{yz}. Two hydrogen atoms are swapped and the px orbital picks up
		# a negative sign. 
		r3=np.zeros([16,16])
		r3[0,0]=1
		r3[1,1]=1
		r3[2,2]=-1
		r3[3,3]=1
		r3[4,4]=1
		r3[5,5]=1
		r3[6,7]=1
		r3[7,6]=1
		r3[8,8]=1
		r3[9,9]=1
		r3[10,10]=-1
		r3[11,11]=1
		r3[12,12]=1
		r3[13,13]=1
		r3[14,15]=1
		r3[15,14]=1
		r_matrices.append(r3)
		######################################
		# The following mtrices commute with Hamiltonian
		# and hence are symmetries, but these cannot be used
		# to taper off qubits.
		theta = -2*np.pi/3
		r4 = np.eye(16)
		r4[5,5]=r4[6,6]=r4[7,7]=0
		r4[13,13]=r4[14,14]=r4[15,15]=0
		r4[5,6]=r4[6,7]=r4[7,5]=1
		# r4[6,5]=r4[7,6]=r4[5,7]=1
		r4[13,14]=r4[14,15]=r4[15,13]=1
		# r4[14,13]=r4[15,14]=r4[13,15]=1
		r4[2,2]=r4[3,3]=r4[10,10]=r4[11,11]=np.cos(theta)
		r4[2,3]=  np.sin(theta)
		r4[3,2]=  -np.sin(theta)
		r4[10,11]=np.sin(theta)
		r4[11,10]=-np.sin(theta)
		r5 = r4.copy()


		theta = np.pi/3
		r4 = np.eye(16,dtype=complex)
		r4[5,5]=r4[6,6]=0
		r4[13,13]=r4[14,14]=0
		r4[5,6]=r4[6,5]=1
		r4[13,14]=r4[14,13]=1
		r4[2,2]=r4[10,10]=np.cos(theta)
		r4[3,3]=r4[11,11]=-np.cos(theta)
		r4[2,3]=  np.sin(theta)
		r4[3,2]=  np.sin(theta)
		r4[10,11]=np.sin(theta)
		r4[11,10]=np.sin(theta)
		if check_r_matrix_flag:
			if bool(check_r_mat(r_matrices,fer_op,one_b,two_b)):
				print('All the above matrices work!')
	#=================================
		# Methane molecule
	#=================================

	elif MoleculeFlag=='CH4':
		# print(MoleculeFlag)
		
		mol.atom=[['C',  (0.0000 ,	0.0000 ,	0.0000 )],    
				['H',  (0.6276 ,	0.6276 ,	0.6276 )],  
				['H',  (0.6276 ,	-0.6276,	-0.6276)],  
				['H',  (-0.6276,	0.6276 ,	-0.6276)],    
				['H',  (-0.6276,	-0.6276,	0.6276 )]]    

		mol.basis='sto-3g'
		mol.build()
		_q_=int_func.qmol_func(mol, atomic=True)
		fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=_q_.two_body_integrals)
		# Spin symmetry:
		r1=np.zeros([18,18])
		for i in range(18):
			if i<8:
				r1[i+9,i]=1.
			else:
				r1[i-9,i]=1.
		
		if check_r_matrix_flag:
			if bool(check_r_mat(r_matrices,fer_op,one_b,two_b)):
				print('All the above matrices work!')

	#=================================
		# Carbon dioxide molecule
	#=================================
	elif MoleculeFlag=='CO2':
		r_matrices = []
		# print(MoleculeFlag)
		mol.atom = [['C',(0., 0., 0.)],
					['O',(-1.1621, 0., 0.)],
					['O',(1.1621, 0., 0.)]]

		mol.basis='sto-3g'
		mol.build()
		_q_=int_func.qmol_func(mol, atomic=True)
		fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=_q_.two_body_integrals)

		# Spin symmetry:
		r1=np.zeros([30,30])
		for i in range(30):
			if i<15:
				r1[i+15,i]=1.
			else:
				r1[i-15,i]=1.
		r_matrices.append(r1)
		
		# Permutation matrix for inversion. \sigma{yz} All the oxygen orbitals are swapped.
		# Oxygen px orbital picks up a negative sign and the px of carbon picks up a 
		# negative sign.
		r2=np.zeros([30,30])
		r2[0,0]=1
		r2[1,1]=1
		r2[3,3]=1
		r2[2,2]=-1
		r2[4,4]=r2[19,19]=1
		r2[5,10]=r2[10,5]=1
		r2[11,6]=r2[6,11]=1
		r2[7,12]=r2[12,7]=-1
		r2[8,13]=r2[13,8]=1
		r2[14,9]=r2[9,14]=1
		r2[15,15]=1
		r2[16,16]=1
		r2[17,17]=-1
		r2[18,18]=1
		r2[20,25]=r2[25,20]=1
		r2[21,26]=r2[26,21]=1
		r2[22,27]=r2[27,22]=-1
		r2[23,28]=r2[28,23]=1
		r2[24,29]=r2[29,24]=1
		r_matrices.append(r2)
		# Permutation matrix for inversion. \sigma{xy} 
		# pz orbitals of oxygen and carbon pick up negative sign
		r3=np.eye(30)
		r3[4,4]=-1
		r3[19,19]=-1
		r3[9,9]=-1
		r3[14,14]=-1
		r3[24,24]=-1
		r3[29,29]=-1
		r_matrices.append(r3)
		# Permutation matrix for inversion. \sigma{xz} 
		# py orbitals of oxygen and carbon pick up negative sign
		r4=np.eye(30)
		r4[3,3]=-1
		r4[18,18]=-1
		r4[8,8]=-1
		r4[13,13]=-1
		r4[23,23]=-1
		r4[28,28]=-1
		r_matrices.append(r4)
		# Permutation matrix for axial symmetry. C_2{x}
		# pz and py orbitals of oxygen and carbon pick up negative sign
		r5=np.eye(30)
		r5[3,3]=-1
		r5[18,18]=-1
		r5[8,8]=-1
		r5[13,13]=-1
		r5[23,23]=-1
		r5[28,28]=-1
		r5[4,4]=-1
		r5[19,19]=-1
		r5[9,9]=-1
		r5[14,14]=-1
		r5[24,24]=-1
		r5[29,29]=-1	
		r_matrices.append(r5)
		# Permutation matrix for axial symmetry. C_2{y} All the oxygen orbitals are swapped.
		# Oxygen px and pz orbital picks up a negative sign and the px and pz of carbon picks up a 
		# negative sign.
		r6=np.zeros([30,30])
		r6[0,0]=r6[1,1]=1
		r6[3,3]=r6[18,18]=1
		r6[17,17]=r6[2,2]=-1
		r6[4,4]=r6[19,19]=-1
		r6[5,10]=r6[10,5]=1
		r6[11,6]=r6[6,11]=1
		r6[7,12]=r6[12,7]=-1
		r6[8,13]=r6[13,8]=1
		r6[14,9]=r6[9,14]=-1
		r6[15,15]=r6[16,16]=1
		r6[20,25]=r6[25,20]=1
		r6[21,26]=r6[26,21]=1
		r6[22,27]=r6[27,22]=-1
		r6[23,28]=r6[28,23]=1
		r6[24,29]=r6[29,24]=-1
		r_matrices.append(r6)
		# Permutation matrix for axial symmetry. C_2{z} All the oxygen orbitals are swapped.
		# Oxygen px and py orbital picks up a negative sign and the px and py of carbon picks up a 
		# negative sign.
		r7=np.zeros([30,30])
		r7[0,0]=r7[1,1]=1
		r7[3,3]=r7[18,18]=-1
		r7[17,17]=r7[2,2]=-1
		r7[4,4]=r7[19,19]=1
		r7[5,10]=r7[10,5]=1
		r7[11,6]=r7[6,11]=1
		r7[7,12]=r7[12,7]=-1
		r7[8,13]=r7[13,8]=-1
		r7[14,9]=r7[9,14]=1
		r7[15,15]=r7[16,16]=1
		r7[20,25]=r7[25,20]=1
		r7[21,26]=r7[26,21]=1
		r7[22,27]=r7[27,22]=-1
		r7[23,28]=r7[28,23]=-1
		r7[24,29]=r7[29,24]=1
		r_matrices.append(r7)

		if check_r_matrix_flag:
			if bool(check_r_mat(r_matrices,fer_op,one_b,two_b)):
				print('All the above matrices work!')
	#=================================
		# Ethyne molecule
	#=================================

	elif MoleculeFlag=='C2H2':
		r_matrices =[]
		print(MoleculeFlag)
		num_particles = 14

		if is_atomic:
			try:
				data = np.load(MoleculeFlag+'.npz')
				data.files
				one_b = data['one_b']
				two_b = data['two_b']
			except IOError:
				mol.atom = [['C',(-0.6000, 0.0000, 0.0000 )], 
					['C',(0.6000, 0.0000, 0.0000 )],
					['H',(-1.6650,0.0000, 0.000 )],
					['H',(1.6650,0.0000, 0.000 )]]
				mol.build()
				_q_=int_func.qmol_func(mol, atomic=True)
				one_b=_q_.one_body_integrals
				two_b=_q_.two_body_integrals
				# np.savez(MoleculeFlag+'_ao.npz',one_b=_q_.one_body_integrals,two_b=_q_.two_body_integrals)
		
			two_body_temp = QMolecule.twoe_to_spin(_q_.mo_eri_ints)
			temp_int = np.einsum('ijkl->ljik', _q_.mo_eri_ints)
			two_body_temp = QMolecule.twoe_to_spin(temp_int)
			mol = gto.M(atom=mol.atom, basis='sto-3g')

			O = get_ovlp(mol)
			X = np.kron(np.identity(2), np.linalg.inv(scipy.linalg.sqrtm(O)))

			fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=two_body_temp)
			fer_op.transform(X)
		else:
			mol.atom = [['C',(-0.6000, 0.0000, 0.0000 )], 
					['C',(0.6000, 0.0000, 0.0000 )],
					['H',(-1.6650,0.0000, 0.000 )],
					['H',(1.6650,0.0000, 0.000 )]]
			mol.build()
			_q_=int_func.qmol_func(mol, atomic=is_atomic)
			one_b=_q_.one_body_integrals
			two_b=_q_.two_body_integrals
			fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=_q_.two_body_integrals)


		one_b = fer_op.h1
		two_b = fer_op.h2

		# Spin symmetry:
		r1=np.zeros([24,24])
		for i in range(24):
			if i<12:
				r1[i+12,i]=1.
			else:
				r1[i-12,i]=1.
		# r_matrices.append(r1)
		# R-matrix for plane of symmetry \sigma_{xy}. Everything remains the same, only pz-orbitals pick up negative sign. 
		r2=np.eye(24)
		r2[4,4]=-1
		r2[9,9]=-1
		r2[16,16]=-1
		r2[21,21]=-1
		# R-matrix for plane of symmetry \sigma_{xz}. Everything remains the same, only py-orbitals pick up negative sign. 
		r3=np.eye(24)
		r3[3,3]=-1
		r3[8,8]=-1
		r3[15,15]=-1
		r3[20,20]=-1
		# R-matrix for plane of symmetry \sigma_{yz}. 
		r4=np.zeros([24,24])
		#Swapping px picks up a negative sign
		r4[2,7]=r4[7,2]=-1
		r4[14,19]=r4[19,14]=-1
		#Swapping 1s, 2s, 2py and 2pz of Carbon
		r4[0,5]=r4[5,0]=1
		r4[1,6]=r4[6,1]=1
		r4[12,17]=r4[17,12]=1
		r4[13,18]=r4[18,13]=1
		r4[3,8]=r4[8,3]=1
		r4[15,20]=r4[20,15]=1
		r4[21,16]=r4[16,21]=1
		r4[9,4]=r4[4,9]=1
		#Swapping 1s of hydrogen	
		r4[10,11]=r4[11,10]=1
		r4[22,23]=r4[23,22]=1

		# R-matrix for axis of symmetry C_2 around y axis. 
		r5=np.zeros([24,24])
		#Swapping px picks up a negative sign
		r5[2,7]=r5[7,2]=-1
		r5[14,19]=r5[19,14]=-1
		r5[21,16]=r5[16,21]=-1
		r5[9,4]=r5[4,9]=-1
		#Swapping 1s, 2s, 2py and 2pz of Carbon
		r5[0,5]=r5[5,0]=1
		r5[1,6]=r5[6,1]=1
		r5[12,17]=r5[17,12]=1
		r5[13,18]=r5[18,13]=1
		r5[3,8]=r5[8,3]=1
		r5[15,20]=r5[20,15]=1
		#Swapping 1s of hydrogen	
		r5[10,11]=r5[11,10]=1
		r5[22,23]=r5[23,22]=1
		# R-matrix for axis of symmetry C_2 around z axis. 
		r6=np.zeros([24,24])
		#Swapping px picks up a negative sign
		r6[2,7]=r6[7,2]=-1
		r6[14,19]=r6[19,14]=-1
		r6[15,20]=r6[20,15]=-1
		r6[3,8]=r6[8,3]=-1
		#Swapping 1s, 2s, 2py and 2pz of Carbon
		r6[0,5]=r6[5,0]=1
		r6[1,6]=r6[6,1]=1
		r6[12,17]=r6[17,12]=1
		r6[13,18]=r6[18,13]=1
		r6[9,4]=r6[4,9]=1
		r6[21,16]=r6[16,21]=1
		#Swapping 1s of hydrogen	
		r6[10,11]=r6[11,10]=1
		r6[22,23]=r6[23,22]=1
		
		# R-matrix for x-axis symmetry. pz and py orbitals of both the carbon atoms pick up a negative sign.
		r7=np.eye(24)	
		r7[3,3]=-1
		r7[4,4]=-1
		r7[8,8]=-1
		r7[9,9]=-1
		r7[16,16]=-1
		r7[15,15]=-1
		r7[20,20]=-1
		r7[21,21]=-1
		r_matrices.append(r7)
		r_matrices.append(r6)
		r_matrices.append(r5)
		r_matrices.append(r2)
		r_matrices.append(r3)
		r_matrices.append(r4)
		r_matrices.append(r1)
		if check_r_matrix_flag:
			if bool(check_r_mat(r_matrices,fer_op,one_b,two_b)):
				print('All the above matrices work!')
	#=================================
		# Ethylene molecule
	#=================================

	elif MoleculeFlag=='C2H4':
		r_matrices =[]
		print(MoleculeFlag)
		try:
			data = np.load(MoleculeFlag+'.npz')
			data.files
			one_b = data['one_b']
			two_b = data['two_b']
		except IOError:
			mol.atom = [['C',( 0.6695, 0.0000 , 0.0000)],
		 	 			['C',(-0.6695, 0.0000 , 0.0000)],
		 	 			['H',( 1.2321, 0.9289 , 0.0000)],
		 	 			['H',( 1.2321, -0.9289, 0.0000)],
		 	 			['H',(-1.2321, 0.9289 , 0.0000)],
		 	 			['H',(-1.2321, -0.9289, 0.0000)]]
			mol.build()
			_q_=int_func.qmol_func(mol, atomic=True)
			one_b=_q_.one_body_integrals
			two_b=_q_.two_body_integrals
			np.savez(MoleculeFlag+'.npz',one_b=_q_.one_body_integrals,two_b=_q_.two_body_integrals)

		fer_op = FermionicOperator(h1=one_b, h2=two_b)
		r_matrices = []
		

		# Spin symmetry:
		r1=np.zeros([28,28])
		for i in range(28):
			if i<14:
				r1[i+14,i]=1.
			else:
				r1[i-14,i]=1.
		# r_matrices.append(r1)
		# R-matrix for plane of symmetry \sigma_{xy}. Everything remains the same, only pz-orbitals pick up negative sign. 
		r2=np.eye(28)
		r2[4,4]=-1
		r2[9,9]=-1
		r2[18,18]=-1
		r2[23,23]=-1
		# R-matrix for plane of symmetry \sigma_{xz}. 
		r3=np.eye(28)
		r3[3,3]=-1
		r3[8,8]=-1
		r3[17,17]=-1
		r3[22,22]=-1
		r3[10,10]=r3[11,11]=r3[12,12]=r3[13,13]=0
		r3[24,24]=r3[25,25]=r3[26,26]=r3[27,27]=0
		r3[10,11]=r3[11,10]=r3[12,13]=r3[13,12]=1
		r3[24,25]=r3[25,24]=r3[26,27]=r3[27,26]=1
		# R-matrix for plane of symmetry \sigma_{yz}. 
		r4=np.zeros([28,28])
		# #Swapping px picks up a negative sign
		r4[2,7]=r4[7,2]=-1
		r4[16,21]=r4[21,16]=-1
		# #Swapping 1s, 2s, 2py and 2pz of Carbon
		r4[0,5]=r4[5,0]=1
		r4[1,6]=r4[6,1]=1
		r4[3,8]=r4[8,3]=1
		r4[4,9]=r4[9,4]=1
		r4[14,19]=r4[19,14]=1
		r4[15,20]=r4[20,15]=1
		r4[17,22]=r4[22,17]=1
		r4[18,23]=r4[23,18]=1
		# #Swapping 1s of hydrogen	
		r4[10,12]=r4[12,10]=1
		r4[24,26]=r4[26,24]=1
		r4[11,13]=r4[13,11]=1
		r4[25,27]=r4[27,25]=1
		
		# R-matrix for axis of symmetry C_2 around y axis. 
		r5=np.zeros([28,28])
		#Swapping px picks up a negative sign
		r5[2,7]=r5[7,2]=-1
		r5[4,9]=r5[9,4]=-1
		r5[16,21]=r5[21,16]=-1
		r5[18,23]=r5[23,18]=-1
		#Swapping 1s, 2s, 2py and 2pz of Carbon
		r5[0,5]=r5[5,0]=1
		r5[1,6]=r5[6,1]=1
		r5[3,8]=r5[8,3]=1
		r5[14,19]=r5[19,14]=1
		r5[15,20]=r5[20,15]=1
		r5[17,22]=r5[22,17]=1
		#Swapping 1s of hydrogen	
		r5[10,12]=r5[12,10]=1
		r5[24,26]=r5[26,24]=1
		r5[11,13]=r5[13,11]=1
		r5[25,27]=r5[27,25]=1
		# R-matrix for axis of symmetry C_2 around z axis. 
		r6=np.zeros([28,28])
		#Swapping px picks up a negative sign
		r6[2,7]=r6[7,2]=-1
		r6[3,8]=r6[8,3]=-1
		r6[16,21]=r6[21,16]=-1
		r6[17,22]=r6[22,17]=-1
		#Swapping 1s, 2s, 2py and 2pz of Carbon
		r6[0,5]=r6[5,0]=1
		r6[1,6]=r6[6,1]=1
		r6[4,9]=r6[9,4]=1
		r6[14,19]=r6[19,14]=1
		r6[15,20]=r6[20,15]=1
		r6[18,23]=r6[23,18]=1
		#Swapping 1s of hydrogen	
		r6[10,13]=r6[13,10]=1
		r6[24,27]=r6[27,24]=1
		r6[11,12]=r6[12,11]=1
		r6[25,26]=r6[26,25]=1
		
		# # R-matrix for x-axis symmetry. pz and py orbitals of both the carbon atoms pick up a negative sign.
		r7=np.eye(28)	
		r7[3,3]=-1
		r7[4,4]=-1
		r7[8,8]=-1
		r7[9,9]=-1
		r7[17,17]=-1
		r7[18,18]=-1
		r7[22,22]=-1
		r7[23,23]=-1
		r7[10,10]=r7[11,11]=r7[12,12]=r7[13,13]=0
		r7[24,24]=r7[25,25]=r7[26,26]=r7[27,27]=0
		r7[10,11]=r7[11,10]=r7[12,13]=r7[13,12]=1
		r7[24,25]=r7[25,24]=r7[26,27]=r7[27,26]=1
		r_matrices.append(r1)
		r_matrices.append(r2)
		r_matrices.append(r3)
		r_matrices.append(r4)
		r_matrices.append(r5)
		r_matrices.append(r6)
		r_matrices.append(r7)
		if check_r_matrix_flag:
			if bool(check_r_mat(r_matrices,fer_op,one_b,two_b)):
				print('All the above matrices work!')
	#=================================
		# Boron trifluoride molecule
	#=================================

	elif MoleculeFlag == 'BF3':
		r_matrices = []
		
		try:
			data = np.load(MoleculeFlag+'.npz')
			one_b = data['one_b']
			two_b = data['two_b']
		except IOError:
			mol.atom = [['B',(0.0000	,0.0000	,0.000000 )], 
						['F',(0.0000	,1.3070	,0.00000 )],
						['F',(1.1319	,-0.6535,	0.000 )],
						['F',(-1.1319	,-0.6535,	0.00)]]
			mol.basis = 'sto-3g'
			mol.build()
			_q_=int_func.qmol_func(mol, atomic=True)
			one_b=_q_.one_body_integrals
			two_b=_q_.two_body_integrals
			np.savez(MoleculeFlag+'.npz',one_b=_q_.one_body_integrals,two_b=_q_.two_body_integrals)

		fer_op = FermionicOperator(h1=one_b, h2=two_b)
		
		
		
		# Spin symmetry:
		r1=np.zeros([40,40])
		for i in range(40):
			if i<20:
				r1[i+20,i]=1.
			else:
				r1[i-20,i]=1.
		r_matrices.append(r1)
		# R-matrix for plane of symmetry \sigma_{xy}. 
		r2=np.eye(40)
		r2[4,4]=-1
		r2[9,9]=-1
		r2[14,14]=-1
		r2[19,19]=-1
		r2[24,24]=-1
		r2[29,29]=-1
		r2[34,34]=-1
		r2[39,39]=-1
		r_matrices.append(r2)

		# R-matrix for plane of symmetry \sigma_{yz}. 
		r3=np.eye(40)
		r3[2,2]=-1
		r3[7,7]=-1
		r3[22,22]=-1
		r3[27,27]=-1
		for i in range(5):
			r3[10+i,10+i]=0
			r3[15+i,15+i]=0
			r3[30+i,30+i]=0
			r3[35+i,35+i]=0
			r3[10+i,15+i]=1
			r3[15+i,10+i]=1
			r3[30+i,35+i]=1
			r3[35+i,30+i]=1
		
		r3[12,17]=-1
		r3[17,12]=-1
		r3[32,37]=-1
		r3[37,32]=-1
		r_matrices.append(r3)
		# R-matrix for axis of symmetry C_2 around y axis. 
		r4=np.eye(40)
		r4[2,2]=-1
		r4[4,4]=-1
		r4[7,7]=-1
		r4[9,9]=-1
		r4[22,22]=-1
		r4[24,24]=-1
		r4[29,29]=-1
		r4[27,27]=-1
		for i in range(5):
			r4[10+i,10+i]=0
			r4[15+i,15+i]=0
			r4[30+i,30+i]=0
			r4[35+i,35+i]=0
			r4[10+i,15+i]=1
			r4[15+i,10+i]=1
			r4[30+i,35+i]=1
			r4[35+i,30+i]=1
		
		r4[12,17]=-1
		r4[17,12]=-1
		r4[14,19]=-1
		r4[19,14]=-1
		r4[32,37]=-1
		r4[37,32]=-1
		r4[34,39]=-1
		r4[39,34]=-1
		r_matrices.append(r4)
		# R-matrix for plane of symmetry \sigma_{yz}. 
		# r4=np.zeros([24,24])
		# #Swapping px picks up a negative sign
		# r4[2,7]=r4[7,2]=-1
		# r4[14,19]=r4[19,14]=-1
		# #Swapping 1s, 2s, 2py and 2pz of Carbon
		# r4[0,5]=r4[5,0]=1
		# r4[1,6]=r4[6,1]=1
		# r4[12,17]=r4[17,12]=1
		# r4[13,18]=r4[18,13]=1
		# r4[3,8]=r4[8,3]=1
		# r4[15,20]=r4[20,15]=1
		# r4[21,16]=r4[16,21]=1
		# r4[9,4]=r4[4,9]=1
		# #Swapping 1s of hydrogen	
		# r4[10,11]=r4[11,10]=1
		# r4[22,23]=r4[23,22]=1


	

		# # R-matrix for axis of symmetry C_2 around z axis. 
		# r6=np.zeros([24,24])
		# #Swapping px picks up a negative sign
		# r6[2,7]=r6[7,2]=-1
		# r6[14,19]=r6[19,14]=-1
		# r6[15,20]=r6[20,15]=-1
		# r6[3,8]=r6[8,3]=-1
		# #Swapping 1s, 2s, 2py and 2pz of Carbon
		# r6[0,5]=r6[5,0]=1
		# r6[1,6]=r6[6,1]=1
		# r6[12,17]=r6[17,12]=1
		# r6[13,18]=r6[18,13]=1
		# r6[9,4]=r6[4,9]=1
		# r6[21,16]=r6[16,21]=1
		# #Swapping 1s of hydrogen	
		# r6[10,11]=r6[11,10]=1
		# r6[22,23]=r6[23,22]=1

		# # R-matrix for x-axis symmetry. pz and py orbitals of both the carbon atoms pick up a negative sign.
		# r7=np.eye(24)
		
		# r7[3,3]=-1
		# r7[4,4]=-1
		# r7[8,8]=-1
		# r7[9,9]=-1
		# r7[16,16]=-1
		# r7[15,15]=-1
		# r7[20,20]=-1
		# r7[21,21]=-1

		# print(r4)
		# print(np.real(fer_op.h1[0:int(np.size(fer_op.h1,0)/2),0:int(np.size(fer_op.h1,0)/2)]))		
		# fer_op.transform(r4)
		# # print(np.real(fer_op.h1[0:int(np.size(fer_op.h1,0)/2),0:int(np.size(fer_op.h1,0)/2)]))		
		# # print(np.real(fer_op.h1[0:int(np.size(fer_op.h1,0)/2),0:int(np.size(fer_op.h1,0)/2)]-_q_.one_body_integrals[0:int(np.size(fer_op.h1,0)/2),0:int(np.size(fer_op.h1,0)/2)]))
		# # print(np.real(fer_op.h1[int(np.size(fer_op.h1,0)/2):int(np.size(fer_op.h1,0)),int(np.size(fer_op.h1,0)/2):int(np.size(fer_op.h1,0))]-_q_.one_body_integrals[int(np.size(fer_op.h1,0)/2):int(np.size(fer_op.h1,0)),int(np.size(fer_op.h1,0)/2):int(np.size(fer_op.h1,0))]))
		# print(MoleculeFlag)
		# # print(np.real(fer_op.h1[0:int(np.size(fer_op.h1,0)/2),0:int(np.size(fer_op.h1,0)/2)]))
		# # print(np.real(fer_op.h1[int(np.size(fer_op.h1,0)/2):int(np.size(fer_op.h1,0)),int(np.size(fer_op.h1,0)/2):int(np.size(fer_op.h1,0))]))
		# # print(np.real(fer_op.h1[0:int(np.size(fer_op.h1,0)/2),0:int(np.size(fer_op.h1,0)/2)]))
		# print(np.all(np.abs(fer_op.h1-_q_.one_body_integrals)<1.e-14))
		# print(np.all(np.abs(fer_op.h2-_q_.two_body_integrals)<1.e-14))
		# print(np.linalg.norm(fer_op.h2-_q_.two_body_integrals))
		if check_r_matrix_flag:
			if bool(check_r_mat(r_matrices,fer_op,one_b,two_b)):
				print('All the above matrices work!')
	
#=================================
	# Lithium hydride
#=================================

	elif MoleculeFlag == "LiH":
		r_matrices =[]
		print(MoleculeFlag)

		if is_atomic:
			try:
				data = np.load(MoleculeFlag+'.npz')
				data.files
				one_b = data['one_b']
				two_b = data['two_b']
			except IOError:
				mol.atom = [['Li',(0.000, 0.0000, 0.0000 )],
					['H',(1.5949,0.0000, 0.000 )]]
				mol.build()
				_q_=int_func.qmol_func(mol, atomic=is_atomic)
				one_b=_q_.one_body_integrals
				two_b=_q_.two_body_integrals
				np.savez(MoleculeFlag+'_ao.npz',one_b=_q_.one_body_integrals,two_b=_q_.two_body_integrals)
		
			two_body_temp = QMolecule.twoe_to_spin(_q_.mo_eri_ints)
			temp_int = np.einsum('ijkl->ljik', _q_.mo_eri_ints)
			two_body_temp = QMolecule.twoe_to_spin(temp_int)
			mol = gto.M(atom=mol.atom, basis='sto-3g')

			O = get_ovlp(mol)
			X = np.kron(np.identity(2), np.linalg.inv(scipy.linalg.sqrtm(O)))
			print(_q_.one_body_integrals)
			fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=two_body_temp)
			fer_op.transform(X)
		else:
			mol.atom = [['Li',(0.000, 0.0000, 0.0000 )],
					['H',(1.5949,0.0000, 0.000 )]]
			mol.build()
			_q_=int_func.qmol_func(mol, atomic=is_atomic)
			one_b=_q_.one_body_integrals
			two_b=_q_.two_body_integrals
			fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=_q_.two_body_integrals)


		one_b = fer_op.h1
		two_b = fer_op.h2

		# Spin symmetry:
		r1=np.zeros([12,12])
		for i in range(12):
			if i<6:
				r1[i+6,i]=1.
			else:
				r1[i-6,i]=1.
		r_matrices.append(r1)
		# sigma{xy}
		r2 = np.eye(12)
		r2[4,4]=-1
		r2[10,10]=-1
		r_matrices.append(r2)

		#sigma(yz)
		r3 = np.eye(12)
		r3[3,3]=-1
		r3[9,9]=-1
		r_matrices.append(r3)
		if bool(check_r_mat(r_matrices,fer_op,one_b,two_b)):
			print('All the above matrices work!')
#=================================
	# Beryllium hydride
#=================================
	
	elif MoleculeFlag == "BeH2":
		num_particles = 6
		r_matrices =[]
		print(MoleculeFlag)
		
		if is_atomic:
			try:
				data = np.load(MoleculeFlag+'.npz')
				data.files
				one_b = data['one_b']
				two_b = data['two_b']
			except IOError:
				mol.atom = [['Be',(0.000, 0.0000, 0.0000 )],
						['H',(1.291,0.0000, 0.000 )],
						['H',(-1.291,0.0000, 0.000 )]]
				mol.build()
				_q_=int_func.qmol_func(mol, atomic=True)
				one_b=_q_.one_body_integrals
				two_b=_q_.two_body_integrals
				# np.savez(MoleculeFlag+'_ao.npz',one_b=_q_.one_body_integrals,two_b=_q_.two_body_integrals)
		
			two_body_temp = QMolecule.twoe_to_spin(_q_.mo_eri_ints)
			temp_int = np.einsum('ijkl->ljik', _q_.mo_eri_ints)
			two_body_temp = QMolecule.twoe_to_spin(temp_int)
			mol = gto.M(atom=mol.atom, basis='sto-3g')

			O = get_ovlp(mol)
			X = np.kron(np.identity(2), np.linalg.inv(scipy.linalg.sqrtm(O)))

			fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=two_body_temp)
			fer_op.transform(X)
		else:
			mol.atom = [['Be',(0.000, 0.0000, 0.0000 )],
						['H',(1.291,0.0000, 0.000 )],
						['H',(-1.291,0.0000, 0.000 )]]
			mol.build()
			_q_=int_func.qmol_func(mol, atomic=is_atomic)
			one_b=_q_.one_body_integrals
			two_b=_q_.two_body_integrals
			fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=_q_.two_body_integrals)


		one_b = fer_op.h1
		two_b = fer_op.h2

		# fer_op = FermionicOperator(h1=one_b, h2=two_b)
		# R-matrix for plane of symmetry \sigma_{xy}. Everything remains the same, only pz-orbitals pick up negative sign.
		r1=np.zeros([14,14])
		r1[0,0]=1
		r1[1,1]=1
		r1[2,2]=1
		r1[3,3]=1
		r1[4,4]=-1
		r1[5,5]=1
		r1[6,6]=1
		r1[7,7]=1
		r1[8,8]=1
		r1[9,9]=1
		r1[10,10]=1
		r1[11,11]=-1
		r1[12,12]=1
		r1[13,13]=1

		r_matrices.append(r1)
		# R-matrix for plane of symmetry \sigma_{xz}. Everything remains the same, only py-orbitals pick up negative sign.
		r2=np.eye(14)
		r2[3,3]=-1
		r2[10,10]=-1
		r_matrices.append(r2)

		# R-matrix for plane of symmetry \sigma_{yz}. Everything remains the same, only px-orbitals pick up negative sign and hydrogen atoms swap.
		r3=np.zeros([14,14])
		r3[0,0]=1
		r3[1,1]=1
		r3[2,2]=-1
		r3[3,3]=1
		r3[4,4]=1
		r3[5,6]=1
		r3[6,5]=1
		r3[7,7]=1
		r3[8,8]=1
		r3[9,9]=-1
		r3[10,10]=1
		r3[11,11]=1
		r3[12,13]=1
		r3[13,12]=1
		r_matrices.append(r3)

		# R-matrix for symmetry-axis C_2. Linear water molecule has three axis of symmetry:
		# About z-axis
		r4=np.zeros([14,14])
		r4[0,0]=1
		r4[1,1]=1
		r4[2,2]=-1
		r4[3,3]=-1
		r4[4,4]=1
		r4[5,6]=1
		r4[6,5]=1
		r4[7,7]=1
		r4[8,8]=1
		r4[9,9]=-1
		r4[10,10]=-1
		r4[11,11]=1
		r4[12,13]=1
		r4[13,12]=1
		# r_matrices.append(r4)
		#About y-axis
		r5=np.zeros([14,14])
		r5[0,0]=1
		r5[1,1]=1
		r5[2,2]=-1
		r5[3,3]=1
		r5[4,4]=-1
		r5[5,6]=1
		r5[6,5]=1
		r5[7,7]=1
		r5[8,8]=1
		r5[9,9]=-1
		r5[10,10]=1
		r5[11,11]=-1
		r5[12,13]=1
		r5[13,12]=1
		# r_matrices.append(r5)
		#Symmetry about x-axis:
		r6=np.zeros([14,14])
		r6[0,0]=1
		r6[1,1]=1
		r6[2,2]=1
		r6[3,3]=-1
		r6[4,4]=-1
		r6[5,5]=1
		r6[6,6]=1
		r6[7,7]=1
		r6[8,8]=1
		r6[9,9]=1
		r6[10,10]=-1
		r6[11,11]=-1
		r6[12,12]=1
		r6[13,13]=1
		# r_matrices.append(r6)
		
		#Spin symmetry:
		r7=np.zeros([14,14])
		for i in range(14):
			if i<7:
				r7[i+7,i]=1.
			else:
				r7[i-7,i]=1.
		r_matrices.append(r7)

		if check_r_matrix_flag and is_atomic:
			if bool(check_r_mat(r_matrices,fer_op,one_b,two_b)):
				print('All the above matrices work!')
		

	elif MoleculeFlag == "test":
		r_matrices =[]
		num_particles = 6
		print(MoleculeFlag)
		# try:
		# 	data = np.load(MoleculeFlag+'.npz')
		# 	data.files
		# 	one_b = data['one_b']
		# 	two_b = data['two_b']
		# except IOError:
		mol.atom = [['Be',(0.000, 0.0000, 0.0000 )],
					['H',(1.3264,0.0000, 0.000 )],
					['H',(-1.3264,0.0000, 0.000 )]]
		mol.atom = [['Be',(0.000, 0.0000, 0.0000 )],
					['H',(1.291,0.0000, 0.000 )],
					['H',(-1.291,0.0000, 0.000 )]]
		mol.atom = [['Be',(0.000, 0.0000, 0.0000 )],
					['H',(1.3,0.0000, 0.000 )],
					['H',(-1.3,0.0000, 0.000 )]]		
		mol.basis = 'sto-6g'					
		# mol.symmetry=True
		# mol.atom = [['H',(0, 0, -0.3707)], ['H',(0,0.0,0.3707)]]		
		mol.build()
		# is_atomic = 
		_q_ = qmol_func(mol, atomic=is_atomic)
		if is_atomic:
			two_body_temp = QMolecule.twoe_to_spin(_q_.mo_eri_ints)
			temp_int = np.einsum('ijkl->ljik', _q_.mo_eri_ints)
			two_body_temp = QMolecule.twoe_to_spin(temp_int)
			mol = gto.M(atom=mol.atom, basis='sto-3g')

			O = get_ovlp(mol)
			X = np.kron(np.identity(2), np.linalg.inv(scipy.linalg.sqrtm(O)))

			fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=two_body_temp)
			fer_op.transform(X)
		else:
			fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=_q_.two_body_integrals)


		one_b = fer_op.h1
		two_b = fer_op.h2
		
			# np.savez(MoleculeFlag+'.npz',one_b=_q_.one_body_integrals,two_b=_q_.two_body_integrals)

		# fer_op = FermionicOperator(h1=one_b, h2=two_b)
		# print(fer_op.h1)
		# exit()
		# r_matrices.append(np.eye(14))
		# R-matrix for plane of symmetry \sigma_{xy}. Everything remains the same, only pz-orbitals pick up negative sign.
		r1=np.zeros([14,14])
		r1[0,0]=1
		r1[1,1]=1
		r1[2,2]=1
		r1[3,3]=1
		r1[4,4]=-1
		r1[5,5]=1
		r1[6,6]=1
		r1[7,7]=1
		r1[8,8]=1
		r1[9,9]=1
		r1[10,10]=1
		r1[11,11]=-1
		r1[12,12]=1
		r1[13,13]=1

		r_matrices.append(r1)
		# R-matrix for plane of symmetry \sigma_{xz}. Everything remains the same, only py-orbitals pick up negative sign.
		r2=np.eye(14)
		r2[3,3]=-1
		r2[10,10]=-1
		r_matrices.append(r2)

		# R-matrix for plane of symmetry \sigma_{yz}. Everything remains the same, only px-orbitals pick up negative sign and hydrogen atoms swap.
		r3=np.zeros([14,14])
		r3[0,0]=1
		r3[1,1]=1
		r3[2,2]=-1
		r3[3,3]=1
		r3[4,4]=1
		r3[5,6]=1
		r3[6,5]=1
		r3[7,7]=1
		r3[8,8]=1
		r3[9,9]=-1
		r3[10,10]=1
		r3[11,11]=1
		r3[12,13]=1
		r3[13,12]=1
		r_matrices.append(r3)

		# R-matrix for symmetry-axis C_2. Linear water molecule has three axis of symmetry:
		# About z-axis
		r4=np.zeros([14,14])
		r4[0,0]=1
		r4[1,1]=1
		r4[2,2]=-1
		r4[3,3]=-1
		r4[4,4]=1
		r4[5,6]=1
		r4[6,5]=1
		r4[7,7]=1
		r4[8,8]=1
		r4[9,9]=-1
		r4[10,10]=-1
		r4[11,11]=1
		r4[12,13]=1
		r4[13,12]=1
		# r_matrices.append(r4)
		#About y-axis
		r5=np.zeros([14,14])
		r5[0,0]=1
		r5[1,1]=1
		r5[2,2]=-1
		r5[3,3]=1
		r5[4,4]=-1
		r5[5,6]=1
		r5[6,5]=1
		r5[7,7]=1
		r5[8,8]=1
		r5[9,9]=-1
		r5[10,10]=1
		r5[11,11]=-1
		r5[12,13]=1
		r5[13,12]=1
		# r_matrices.append(r5)
		#Symmetry about x-axis:
		r6=np.zeros([14,14])
		r6[0,0]=1
		r6[1,1]=1
		r6[2,2]=1
		r6[3,3]=-1
		r6[4,4]=-1
		r6[5,5]=1
		r6[6,6]=1
		r6[7,7]=1
		r6[8,8]=1
		r6[9,9]=1
		r6[10,10]=-1
		r6[11,11]=-1
		r6[12,12]=1
		r6[13,13]=1
		# r_matrices.append(r6)
		
		#Spin symmetry:
		r7=np.zeros([14,14])
		for i in range(14):
			if i<7:
				r7[i+7,i]=1.
			else:
				r7[i-7,i]=1.
		# r_matrices.append(r7)
		
		r8=np.zeros([14,14])
		r8[0,0]=1
		r8[1,1]=1
		r8[2,2]=-1
		r8[3,3]=-1
		r8[4,4]=-1
		r8[5,6]=1
		r8[6,5]=1
		r8[7,7]=1
		r8[8,8]=1
		r8[9,9]=-1
		r8[10,10]=-1
		r8[11,11]=-1
		r8[12,13]=1
		r8[13,12]=1
		# r_matrices.append(r8)
		# r_matrices=[]
		# r_matrices.append(np.eye(fer_op.modes))
		if check_r_matrix_flag:
			if bool(check_r_mat(r_matrices,fer_op,one_b,two_b)):
				print('All the above matrices work!')
	if is_atomic:
		return [r_matrices,fer_op,num_particles]
	else:
		r_matrices =[]
		r_matrices.append(np.eye(fer_op.modes))
		return [r_matrices,fer_op, num_particles]

def check_r_mat(r_matrices, fer_op, one_b, two_b):
		r_true = 0
		etol = 1.e-5
		for r in r_matrices:
			temp_fo = copy.deepcopy(fer_op)
			temp_fo.transform(r)
			ts = np.shape(temp_fo.h1)
			# if np.all(np.abs(temp_fo.h1-one_b)<etol) and np.all(np.abs(temp_fo.h2-two_b)<etol) and np.linalg.norm(temp_fo.h2-two_b)<etol:
			if np.all(np.abs(temp_fo.h1-one_b)<etol) and np.all(np.abs(temp_fo.h2-two_b)<etol):
				r_true+=1
				print('True')
			else:
				print('The r-matrix does not commute with the Hamiltonian')
				print(r.real)
				print(np.real(one_b))
				print(np.real(temp_fo.h1))
				print((temp_fo.h1-one_b).real)
				print(np.linalg.norm(temp_fo.h2-two_b))
				# print(two_b[np.where((temp_fo.h2-two_b)>1.e-6)])
				# print((temp_fo.h2[np.where((temp_fo.h2-two_b)>1.e-6)]-two_b[np.where((temp_fo.h2-two_b)>1.e-6)]).real*1000000)
				# print(temp_fo.h1[0:int(ts[0]/2),0:int(ts[0]/2)]-one_b[0:int(ts[0]/2),0:int(ts[0]/2)])
				print("The above r matrix does not work!")
		# if r_true!=len(r_matrices):
			# print(fer_op.h1)
		return (r_true==len(r_matrices))

def check_commute(op1, op2):
	op3 = op1 * op2 - op2 * op1
	op3.chop()
# 	op3.zeros_coeff_elimination()
	return op3.is_empty()

if __name__ == '__main__':
    import sys
	
    if len(sys.argv)!=0:
	    # print(sys.argv[1])
	    [r_mat,fer,num_part]=mol_r_matrices(sys.argv[1],True,True)

		
    # print(np.all(np.matmul(r_mat[1],r_mat[2])==r_mat[0]))
# print("after __name__ guard")