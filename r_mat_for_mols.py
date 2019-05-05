# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 14:46:51 2018

@author: kanav
"""
import logging
from pyscf import gto, scf, ao2mo
from pyscf.lib import param
from scipy import linalg as scila
from pyscf.lib import logger as pylogger
# from qiskit.chemistry import AquaChemistryError
from qiskit.chemistry import QMolecule
import numpy as np
# import gse_algo as ga

from qiskit.chemistry import FermionicOperator
logger = logging.getLogger(__name__)
import int_func

np.set_printoptions(linewidth=230,suppress=True,precision=3,threshold=5000)

MoleculeFlag = 'NH3'
mol = gto.Mole()

if MoleculeFlag == 'H2':
	
	#=================================
	# Hydrogen molecule
	#=================================
	mol = gto.M(atom='H 0 0 0; H 0 0 .7414', basis='sto-3g')
	mol.build()
	_q_=int_func.qmol_func(mol, atomic=True)
	fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=_q_.two_body_integrals)

	# Defining R-matrix --> r
	# Swapping the spatial orbitals. This involves swapping both the spin orbitals corresponding to a spatial orbital.
	r = np.zeros([4,4])
	# r[0,1]=1
	# r[1,0]=1
	# r[2,3]=1
	# r[3,2]=1

	# Swapping the spin oritals. Spin symmetry.
	# r=np.zeros(4,4)
	for i in range(4):
	   if i<2:
	       r[i+2,i]=1.
	   else:
	       r[i-2,i]=1.

	print(np.all(np.abs(fer_op.h1-_q_.one_body_integrals)<1.e-14))
	print(np.all(np.abs(fer_op.h2-_q_.two_body_integrals)<1.e-14))
	fer_op.transform(r)
	print(MoleculeFlag)
	print(np.all(np.abs(fer_op.h1-_q_.one_body_integrals)<1.e-14))
	print(np.all(np.abs(fer_op.h2-_q_.two_body_integrals)<1.e-14))
	print(np.linalg.norm(fer_op.h2-_q_.two_body_integrals))
elif MoleculeFlag== 'H2O_L':
	# print(MoleculeFlag)
	#=================================
	# Water molecule (with different basis sets)
	#=================================
	# Configuration from 
	# mol.atom = [['O',(0.8638, 0.4573,0.0)], ['H',(0, 0, 0)], ['H',(1.7785,0.0,0.0)]]
	mol.atom = [['O',(0.0, 0.0,0.0)],['H',(1, 0, 0)], ['H',(-1.0,0.0,0.0)]]
	#mol.atom = [['O',(0, 0, 0)], ['H',(0, 1, 0)], ['H@2',(0, 0, 1)]]
	mol.basis = 'sto-3g'
	#mol.basis = {'O': 'sto-3g', 'H': 'cc-pvdz', 'H@2': '6-31G'}
	#mol.basis ={'O': gto.basis.parse('''

	mol.build()
	_q_=int_func.qmol_func(mol, atomic=True)
	fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=_q_.two_body_integrals)
	# R-matrix for plane of symmetry \sigma_{xy}. Everything remains the same, only pz-orbitals pick up negative sign.
	r=np.zeros([14,14])
	# r[0,0]=1
	# r[1,1]=1
	# r[2,2]=1
	# r[3,3]=1
	# r[4,4]=-1
	# r[5,5]=1
	# r[6,6]=1
	# r[7,7]=1
	# r[8,8]=1
	# r[9,9]=1
	# r[10,10]=1
	# r[11,11]=-1
	# r[12,12]=1
	# r[13,13]=1

	# R-matrix for plane of symmetry \sigma_{xz}. Everything remains the same, only py-orbitals pick up negative sign.
	# r=np.eye(14)
	# r[3,3]=-1
	# r[10,10]=-1

	# R-matrix for plane of symmetry \sigma_{yz}. Everything remains the same, only px-orbitals pick up negative sign and hydrogen atoms swap.
	# r=np.zeros([14,14])
	# r[0,0]=1
	# r[1,1]=1
	# r[2,2]=-1
	# r[3,3]=1
	# r[4,4]=1
	# r[5,6]=1
	# r[6,5]=1
	# r[7,7]=1
	# r[8,8]=1
	# r[9,9]=-1
	# r[10,10]=1
	# r[11,11]=1
	# r[12,13]=1
	# r[13,12]=1


	# R-matrix for symmetry-axis C_2. Linear water molecule has three axis of symmetry:
	# About z-axis
	# r=np.zeros([14,14])
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
	#About y-axis
	# r=np.zeros([14,14])
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
	#Symmetry about x-axis:
	r=np.zeros([14,14])
	r[0,0]=1
	r[1,1]=1
	r[2,2]=1
	r[3,3]=-1
	r[4,4]=-1
	r[5,5]=1
	r[6,6]=1
	r[7,7]=1
	r[8,8]=1
	r[9,9]=1
	r[10,10]=-1
	r[11,11]=-1
	r[12,12]=1
	r[13,13]=1
	
	#Spin symmetry:
	# for i in range(14):
	#    if i<7:
	#        r[i+7,i]=1.
	#    else:
	#        r[i-7,i]=1.

	fer_op.transform(r)
	# print(_q_._one_body_integrals)
	print(MoleculeFlag)
	print(np.all(np.abs(fer_op.h1-_q_.one_body_integrals)<1.e-16))
	print(np.all(np.abs(fer_op.h2-_q_.two_body_integrals)<1.e-16))
	print(np.linalg.norm(fer_op.h2-_q_.two_body_integrals))
elif MoleculeFlag== 'H2O':
	# print(MoleculeFlag)
	#=================================
	# Water molecule (with different basis sets)
	#=================================
	# Configuration from 
	mol.atom = [['O',(0.0000, 0.0000, 0.0000)],
				['H',(0.757, 0.586, 0.0)],
				['H',(-0.757, 0.586, 0.0)]]

	#mol.atom = [['O',(0, 0, 0)], ['H',(0, 1, 0)], ['H@2',(0, 0, 1)]]
	mol.basis = 'sto-3g'
	#mol.basis = {'O': 'sto-3g', 'H': 'cc-pvdz', 'H@2': '6-31G'}
	mol.build()
	_q_=int_func.qmol_func(mol, atomic=True)
	fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=_q_.two_body_integrals)
	

	#Spin symmetry:
	# for i in range(14):
	#    if i<7:
	#        r[i+7,i]=1.
	#    else:
	#        r[i-7,i]=1.

	# R-matrix for plane of symmetry \sigma_{xy}. Everything remains the same, only pz-orbitals pick up negative sign.
	r=np.eye(14)
	r[4,4]=-1
	r[11,11]=-1
	
	# R-matrix for plane of symmetry \sigma_{yz}. Everything remains the same, only px-orbitals pick up negative sign and the hydrogen atoms swap.
	r=np.eye(14)
	r[2,2]=-1
	r[9,9]=-1
	r[12,12]=0
	r[13,13]=0
	r[12,13]=1
	r[13,12]=1
	r[5,6]=1
	r[6,5]=1
	r[5,5]=0
	r[6,6]=0
	print(r)

	#Axial symmetry about y-axis
	# r=np.zeros([14,14])
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


	fer_op.transform(r)
	# print(_q_._one_body_integrals)
	print(MoleculeFlag)
	print(np.all(np.abs(fer_op.h1-_q_.one_body_integrals)<1.e-14))
	print(np.all(np.abs(fer_op.h2-_q_.two_body_integrals)<1.e-14))
	print(np.linalg.norm(fer_op.h2-_q_.two_body_integrals))
elif MoleculeFlag=='NH3':
	print(MoleculeFlag)
	#=================================
	# Ammonia molecule
	#=================================

	# mol.atom = [['N', (0.0000,  0.0000, 0.0000)],   
	# 			['H', (-0.4417, 0.2906, 0.8711)],  
	# 			['H', (0.7256,  0.6896,-0.1907)],  
	# 			['H', (0.4875, -0.8701, 0.2089)]]
	mol.atom = [['N', (0.0000,  0.0000, 0.0000)],   
				['H', (0.0000,	-0.9377,-0.3816)],  
				['H', (0.8121,	0.4689	,-0.3816)],  
				['H', (-0.8121,	0.4689	,-0.3816)]]

	# mol.atom = [['N', (0.0000,  0.0000, 0.0000)],   
	# 			['H', (-0.663054, -0.663054, -0.3816)],  
	# 			['H', ( 0.905804, -0.242679, -0.3816)],  
	# 			['H', (-0.242679,  0.905804, -0.3816)]]
	

	# mol.atom = [['N', (0.0000,  0.0000, 0.0000)],  
	# 			['H', (-0.8121,	0.4689	,-0.3816)],   
	# 			['H', (0.0000,	-0.9377,	-0.3816)],  
	# 			['H', (0.8121,	0.4689	,-0.3816)]]

	# theta = 2*np.pi/3
	# rot_mat_vec = np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta), np.cos(theta),0],[0,0,1]])
	# R3 = np.eye(16)
	# Id = np.eye(16)
	# R3[2:5,2:5]=rot_mat_vec
	
	mol.symmetry=True
	# mol.symmetry_subgroup
	mol.basis='sto-3g'
	# mol.basis = {'N': 'cc-pvdz', 'H': 'sto-3g'}
	mol.build()
	_q_=int_func.qmol_func(mol, atomic=True)
	fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=_q_.two_body_integrals)
	# fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=np.zeros())
	# from tempfile import TemporaryFile
	# outfile = open('outfile.txt','w+')
	np.savez('outfile',fer_op.h1)
	fer_op_1 = np.load('outfile.npz')
	# print((1+R3+np.dot(R3,R3))*fer_op.h1*(1+R3+np.dot(R3,R3)))
	# print(rot_mat_vec)
	
	# print(fer_op.h1==fer_op_1['arr_0'])
	# print(fer_op.h1)
	# print(fer_op_1['arr_0'])
	# if fer_op.h1==fer_op_1:
	# 	print('yes! it is!')

	
	#Spin symmetry:
	# r=np.zeros([16,16])
	# for i in range(16):
	#    if i<8:
	#        r[i+8,i]=1.
	#    else:
	#        r[i-8,i]=1.

	# R-matrix for plane of symmetry \sigma_{xy}. Everything remains the same, only pz-orbitals pick up negative sign. 
	# This is present only in the 2d configuration of the molecule. (worked with the first geometry)
	# r=np.zeros([16,16])
	# r[0,0]=1
	# r[1,1]=1
	# r[2,2]=1
	# r[3,3]=1
	# r[4,4]=-1
	# r[5,5]=1
	# r[6,6]=1
	# r[7,7]=1
	# r[8,8]=1
	# r[9,9]=1
	# r[10,10]=1
	# r[11,11]=1
	# r[12,12]=-1
	# r[13,13]=1
	# r[14,14]=1
	# r[15,15]=1

	# R-matrix for plane of symmetry \sigma_{yz}. Everything remains the same, only px-orbitals pick up negative sign. 
	r=np.zeros([16,16])
	r[0,0]=1
	r[1,1]=1
	r[2,2]=-1
	r[3,3]=1
	r[4,4]=1
	r[5,5]=1
	r[6,7]=1
	r[7,6]=1
	r[8,8]=1
	r[9,9]=1
	r[10,10]=-1
	r[11,11]=1
	r[12,12]=1
	r[13,13]=1
	r[14,15]=1
	r[15,14]=1


	# R-matrix for plane of symmetry \sigma_{xy}. Everything remains the same, only pz-orbitals pick up negative sign. 
	# This is present only in the 2d configuration of the molecule.
	# r=np.zeros([16,16])
	# r[0,0]=1
	# r[1,1]=1
	# r[2,2]=1/2
	# # r[2,3]=np.sqrt(3)/2
	# r[3,3]=1/2
	# r[3,2]=-np.sqrt(3)/2
	# r[4,4]=1
	# r[5,6]=1
	# r[6,7]=1
	# r[7,5]=1
	# r[8,8]=1
	# r[9,9]=1
	# r[10,10]=1/2
	# r[10,11]=np.sqrt(3)/2
	# r[11,11]=1/2
	# r[11,10]=-np.sqrt(3)/2
	# r[12,12]=1
	# r[13,14]=1
	# r[14,15]=1
	# r[15,13]=1
	
	
	
	c=(1+np.cos(np.pi/3)+np.cos(2*np.pi/3))/np.sqrt(3)
	s=(1+np.sin(np.pi/3)+np.sin(2*np.pi/3))/np.sqrt(3)
	r=np.eye(16)
	
	r[2,2]=c
	r[2,3]=s
	r[3,2]=-s
	r[3,3]=c
	print(r)
	# fer_op.transform(r)

	# r=np.zeros([16,16],dtype=np.complex_)
	# r[0,0]=1
	# r[1,1]=1
	# # r[2,2]=1/np.sqrt(2)
	# # r[2,3]=1/np.sqrt(2) #np.exp(1j*2*np.pi/3)
	# # r[3,2]=-1/np.sqrt(2) #np.exp(1j*2*np.pi/3)
	# # r[3,3]=1/np.sqrt(2)
	# r[2,2]=1
	# r[3,3]=1
	# # r[3,2]=-np.sqrt(3)/2
	# r[4,4]=1
	# # r[5,5]=1/np.sqrt(3)
	# # r[5,6]=1/np.sqrt(3)
	# # r[5,7]=1/np.sqrt(3)
	# # r[6,5]=-1/np.sqrt(3)
	# # r[6,6]=1/np.sqrt(3)
	# # r[6,7]=1/np.sqrt(3)
	# # r[7,5]=1/np.sqrt(3)
	# # r[7,6]=-1/np.sqrt(3)
	# # r[7,7]=1/np.sqrt(3)
	# r[5,5]=1/np.sqrt(2)
	# r[5,6]=-1/np.sqrt(2)
	# # r[5,7]=1/np.sqrt(3)
	# # r[6,5]=-1/np.sqrt(3)
	# r[6,6]=1/np.sqrt(2)
	# r[6,7]=-1/np.sqrt(2)
	# r[7,5]=-1/np.sqrt(2)
	# # r[7,6]=-1/np.sqrt(3)
	# r[7,7]=1/np.sqrt(2)
	# # r[5,6]=1
	# # r[6,7]=1
	# # r[7,5]=1
	# r[8,8]=1
	# r[9,9]=1
	# r[10,10]=1 #np.exp(1j*2*np.pi/3)
	# # r[10,11]=np.sqrt(3)/2
	# r[11,11]=1 #np.exp(1j*2*np.pi/3)
	# # r[11,10]=-np.sqrt(3)/2
	# r[12,12]=1
	# r[13,14]=1
	# r[14,15]=1
	# r[15,13]=1
	# print(r)
	
	
	# x=(np.eye(3)+rot_mat_vec+np.matmul(rot_mat_vec,rot_mat_vec))/3
	# print(np.matmul(x.transpose(),x))
	# print((np.eye(3)+rot_mat_vec+np.matmul(rot_mat_vec,rot_mat_vec)))
	print(np.real(fer_op.h1[0:int(np.size(fer_op.h1,0)/2),0:int(np.size(fer_op.h1,0)/2)]))
	
	# fer_op.transform(1/np.sqrt(3)*(1+R3+np.dot(R3,R3)))
	# print(np.matmul(R3,R3)+R3)
	# print(np.matmul(R3.transpose(),R3))
	fer_op.transform(r)
	print(np.real(fer_op.h1[0:int(np.size(fer_op.h1,0)/2),0:int(np.size(fer_op.h1,0)/2)]))
	# print(rot_mat_vec)
	# print(_q_.one_body_integrals)
	# print(MoleculeFlag)
	# print(np.real(fer_op.h1))
	# print(np.real(fer_op.h1-_q_.one_body_integrals))
	# print(np.matmul(rot_mat_vec,np.array([0.0000,	-0.9377,-0.3816])))
	# print(np.matmul(rot_mat_vec,np.array([0.8121,	0.4689	,-0.3816])))
	# print(np.matmul(rot_mat_vec,np.array([-0.8121,	0.4689	,-0.3816])))

	# print(np.all(np.abs(fer_op.h1-_q_.one_body_integrals)<1.e-14))
	# print(np.all(np.abs(fer_op.h2-_q_.two_body_integrals)<1.e-14))
	# print(np.linalg.norm(fer_op.h2-_q_.two_body_integrals))

	

elif MoleculeFlag=='CH4':
	# print(MoleculeFlag)
	#=================================
	# Methane molecule
	#=================================
	# mol.atom=[['C',  (2.5369,    0.0000,    0.0000)],    
	        #   ['H',  (3.0739,    0.3100,    0.0000)],  
	        #   ['H',  (2.0000,   -0.3100,    0.0000)],  
	        #   ['H',  (2.2269,    0.5369,    0.0000)],    
	        #   ['H',  (2.8469,   -0.5369,    0.0000)]]    

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
	r=np.zeros([18,18])
	for i in range(18):
	   if i<8:
	       r[i+9,i]=1.
	   else:
	       r[i-9,i]=1.
	
	fer_op.transform(r)
	# print(_q_._one_body_integrals.round(3))
	print(MoleculeFlag)
	print(np.all(np.abs(fer_op.h1-_q_.one_body_integrals)<1.e-14))
	print(np.all(np.abs(fer_op.h2-_q_.two_body_integrals)<1.e-14))
	print(np.linalg.norm(fer_op.h2-_q_.two_body_integrals))

elif MoleculeFlag=='O2':
	# print(MoleculeFlag)
	#=================================
	# Oxygen molecule
	#=================================
	mol.atom =[['O', (-1.0000,    0.0000,    0.0000)],
	          ['O', (1.0000,    0.0000,    0.0000)]] 
	mol.basis = 'sto-3g'

	mol.build()
	_q_=int_func.qmol_func(mol, atomic=True)
	fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=_q_.two_body_integrals)
	# Spin symmetry:
	r=np.zeros([10,10])
	for i in range(10):
	   if i<5:
	       r[i+5,i]=1.
	   else:
	       r[i-5,i]=1.



	fer_op.transform(r)
	# print(_q_._one_body_integrals.round(3))
	print(MoleculeFlag)
	print(np.all(np.abs(fer_op.h1-_q_.one_body_integrals)<1.e-14))
	print(np.all(np.abs(fer_op.h2-_q_.two_body_integrals)<1.e-14))
	print(np.linalg.norm(fer_op.h2-_q_.two_body_integrals))
elif MoleculeFlag=='CO2':
	# print(MoleculeFlag)
	#=================================
	# Carbon dioxide molecule
	#=================================

	mol.atom = [['C',(0., 0., 0.)],
	            ['O',(-1.1621, 0., 0.)],
	            ['O',(1.1621, 0., 0.)]]

	mol.basis='sto-3g'
	mol.build()
	_q_=int_func.qmol_func(mol, atomic=True)
	fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=_q_.two_body_integrals)

	# Spin symmetry:
	r=np.zeros([30,30])
	for i in range(30):
	   if i<15:
	       r[i+15,i]=1.
	   else:
	       r[i-15,i]=1.

	
	# Permutation matrix for inversion. \sigma{yz} All the oxygen orbitals are swapped.
	# Oxygen px orbital picks up a negative sign and the px of carbon picks up a 
	# negative sign.
	r=np.zeros([30,30])
	r[0,0]=r[1,1]=1
	r[3,3]=r[2,2]=1
	r[4,4]=r[19,19]=1
	r[5,10]=r[10,5]=1
	r[11,6]=r[6,11]=1
	r[7,12]=r[12,7]=-1
	r[8,13]=r[13,8]=1
	r[14,9]=r[9,14]=1
	r[15,15]=r[16,16]=1
	r[17,17]=r[18,18]=-1
	r[20,25]=r[25,20]=1
	r[21,26]=r[26,21]=1
	r[22,27]=r[27,22]=-1
	r[23,28]=r[28,23]=1
	r[24,29]=r[29,24]=1

	# Permutation matrix for inversion. \sigma{xy} 
	# pz orbitals of oxygen and carbon pick up negative sign
	r=np.eye(30)
	r[4,4]=-1
	r[19,19]=-1
	r[9,9]=-1
	r[14,14]=-1
	r[24,24]=-1
	r[29,29]=-1

	# Permutation matrix for inversion. \sigma{xz} 
	# pz orbitals of oxygen and carbon pick up negative sign
	r=np.eye(30)
	r[3,3]=-1
	r[18,18]=-1
	r[8,8]=-1
	r[13,13]=-1
	r[23,23]=-1
	r[28,28]=-1

	# Permutation matrix for axial symmetry. C_2{x}
	# pz and py orbitals of oxygen and carbon pick up negative sign
	r=np.eye(30)
	r[3,3]=-1
	r[18,18]=-1
	r[8,8]=-1
	r[13,13]=-1
	r[23,23]=-1
	r[28,28]=-1
	r[4,4]=-1
	r[19,19]=-1
	r[9,9]=-1
	r[14,14]=-1
	r[24,24]=-1
	r[29,29]=-1	
	
	# Permutation matrix for axial symmetry. C_2{y} All the oxygen orbitals are swapped.
	# Oxygen px orbital picks up a negative sign and the px of carbon picks up a 
	# negative sign.
	r=np.zeros([30,30])
	r[0,0]=r[1,1]=1
	r[3,3]=r[18,18]=1
	r[17,17]=r[2,2]=-1
	r[4,4]=r[19,19]=-1
	r[5,10]=r[10,5]=1
	r[11,6]=r[6,11]=1
	r[7,12]=r[12,7]=-1
	r[8,13]=r[13,8]=1
	r[14,9]=r[9,14]=-1
	r[15,15]=r[16,16]=1
	r[20,25]=r[25,20]=1
	r[21,26]=r[26,21]=1
	r[22,27]=r[27,22]=-1
	r[23,28]=r[28,23]=1
	r[24,29]=r[29,24]=-1

	# Permutation matrix for axial symmetry. C_2{z} All the oxygen orbitals are swapped.
	# Oxygen px orbital picks up a negative sign and the px of carbon picks up a 
	# negative sign.
	r=np.zeros([30,30])
	r[0,0]=r[1,1]=1
	r[3,3]=r[18,18]=-1
	r[17,17]=r[2,2]=-1
	r[4,4]=r[19,19]=1
	r[5,10]=r[10,5]=1
	r[11,6]=r[6,11]=1
	r[7,12]=r[12,7]=-1
	r[8,13]=r[13,8]=-1
	r[14,9]=r[9,14]=1
	r[15,15]=r[16,16]=1
	r[20,25]=r[25,20]=1
	r[21,26]=r[26,21]=1
	r[22,27]=r[27,22]=-1
	r[23,28]=r[28,23]=-1
	r[24,29]=r[29,24]=1


	##print(r)
	##print(np.all(r==r.transpose()))
	fer_op.transform(r)
	# print(_q_._one_body_integrals.round(3))
	print(MoleculeFlag)
	print(np.all(np.abs(fer_op.h1-_q_.one_body_integrals)<1.e-14))
	print(np.all(np.abs(fer_op.h2-_q_.two_body_integrals)<1.e-14))
	print(np.linalg.norm(fer_op.h2-_q_.two_body_integrals))


elif MoleculeFlag=='C2H2':
	# print(MoleculeFlag)
	#=================================
	# Ethyne molecule
	#=================================

	mol.atom = [['C',(-0.6000, 0.0000, 0.0000 )], 
			    ['C',(0.6000, 0.0000, 0.0000 )],
			    ['H',(-1.6650,0.0000, 0.000 )],
			    ['H',(1.6650,0.0000, 0.000 )]]
	
	mol.build()
	_q_=int_func.qmol_func(mol, atomic=True)
	fer_op = FermionicOperator(h1=_q_.one_body_integrals, h2=_q_.two_body_integrals)


	# Spin symmetry:
	# r=np.zeros([24,24])
	# for i in range(24):
	#    if i<12:
	#        r[i+12,i]=1.
	#    else:
	#        r[i-12,i]=1.

	# R-matrix for plane of symmetry \sigma_{xy}. Everything remains the same, only pz-orbitals pick up negative sign. 
	# r=np.eye(24)
	# r[4,4]=-1
	# r[9,9]=-1
	# r[16,16]=-1
	# r[21,21]=-1

	# R-matrix for plane of symmetry \sigma_{xz}. Everything remains the same, only py-orbitals pick up negative sign. 
	# r=np.eye(24)
	# r[3,3]=-1
	# r[8,8]=-1
	# r[15,15]=-1
	# r[20,20]=-1

	# R-matrix for plane of symmetry \sigma_{yz}. 
	r=np.zeros([24,24])
	#Swapping px picks up a negative sign
	r[2,7]=r[7,2]=-1
	r[14,19]=r[19,14]=-1
	#Swapping 1s, 2s, 2py and 2pz of Carbon
	r[0,5]=r[5,0]=1
	r[1,6]=r[6,1]=1
	r[12,17]=r[17,12]=1
	r[13,18]=r[18,13]=1
	r[3,8]=r[8,3]=1
	r[15,20]=r[20,15]=1
	r[21,16]=r[16,21]=1
	r[9,4]=r[4,9]=1
	#Swapping 1s of hydrogen	
	r[10,11]=r[11,10]=1
	r[22,23]=r[23,22]=1


	# R-matrix for axis of symmetry C_2 around y axis. 
	r=np.zeros([24,24])
	#Swapping px picks up a negative sign
	r[2,7]=r[7,2]=-1
	r[14,19]=r[19,14]=-1
	r[21,16]=r[16,21]=-1
	r[9,4]=r[4,9]=-1
	#Swapping 1s, 2s, 2py and 2pz of Carbon
	r[0,5]=r[5,0]=1
	r[1,6]=r[6,1]=1
	r[12,17]=r[17,12]=1
	r[13,18]=r[18,13]=1
	r[3,8]=r[8,3]=1
	r[15,20]=r[20,15]=1
	#Swapping 1s of hydrogen	
	r[10,11]=r[11,10]=1
	r[22,23]=r[23,22]=1

	# R-matrix for axis of symmetry C_2 around z axis. 
	r=np.zeros([24,24])
	#Swapping px picks up a negative sign
	r[2,7]=r[7,2]=-1
	r[14,19]=r[19,14]=-1
	r[15,20]=r[20,15]=-1
	r[3,8]=r[8,3]=-1
	#Swapping 1s, 2s, 2py and 2pz of Carbon
	r[0,5]=r[5,0]=1
	r[1,6]=r[6,1]=1
	r[12,17]=r[17,12]=1
	r[13,18]=r[18,13]=1
	r[9,4]=r[4,9]=1
	r[21,16]=r[16,21]=1
	#Swapping 1s of hydrogen	
	r[10,11]=r[11,10]=1
	r[22,23]=r[23,22]=1

	# R-matrix for x-axis symmetry. pz and py orbitals of both the carbon atoms pick up a negative sign.
	r=np.eye(24)
	
	r[3,3]=-1
	r[4,4]=-1
	r[8,8]=-1
	r[9,9]=-1
	r[16,16]=-1
	r[15,15]=-1
	r[20,20]=-1
	r[21,21]=-1


	fer_op.transform(r)
	# print(_q_._one_body_integrals.round(3))
	print(MoleculeFlag)
	print(np.all(np.abs(fer_op.h1-_q_.one_body_integrals)<1.e-14))
	print(np.all(np.abs(fer_op.h2-_q_.two_body_integrals)<1.e-14))
	print(np.linalg.norm(fer_op.h2-_q_.two_body_integrals))


# fer_op = FermionicOperator(h1=_q_._one_body_integrals, h2=_q_._two_body_integrals)
# # fer_op_bk = FermionicOperator(h1=_q_._one_body_integrals, h2=np.einsum('ijkl->ikjl',two_body)*-1.)
# fer_op_bk = FermionicOperator(h1=_q_._one_body_integrals, h2=two_body)
# jw_qo=fer_op.mapping('jordan_wigner')
# bk_qo=ga.bksf_mapping(fer_op_bk)
# # print(bk_qo.print_operators())
# bk_qo.to_matrix()
# bksf_eigs = np.linalg.eigvalsh(bk_qo._matrix.todense())
# jw_qo.to_matrix()
# jw_eigs=np.linalg.eigvalsh(jw_qo._matrix.todense())
# jw_eigs=jw_eigs.round(6)
# bksf_eigs=bksf_eigs.round(6)
# print(jw_eigs)
# print(bksf_eigs)
# evensector_H=0
# for i in range(np.size(jw_eigs)):
#     if bool(np.size(np.where(jw_eigs[i] ==
#                                    bksf_eigs))):
#         print(jw_eigs[i])
#         evensector_H += 1


#import symmetric_qubit_reduction as sqr
#si=[]
#r=np.zeros([14,14])
#r[0,0]=1
#r[1,1]=1
#r[2,2]=1
#r[3,3]=1
#r[4,4]=-1
#r[5,5]=1
#r[6,6]=1
#r[7,7]=1
#r[8,8]=1
#r[9,9]=1
#r[10,10]=1
#r[11,11]=-1
#r[12,12]=1
#r[13,13]=1



#print(r)
#fer_op.transform(r)
#print(np.all(np.abs(fer_op.h1-_q_._one_body_integrals)<1.e-16))
#print(np.all(np.abs(fer_op.h2-_q_._two_body_integrals)<1.e-16))
#print(np.linalg.norm(fer_op.h2-_q_._two_body_integrals))

#
#g_matrix = -1j * scila.logm(r)
##print(g_matrix.real.round(3))
#eigval_v,eigvec_v=np.linalg.eig(g_matrix)
##print(eigvec_v,3)
#print(np.dot(eigvec_v.conjugate().transpose(),eigvec_v).round(2))

#s=np.dot(np.dot(eigvec_v.transpose(),g_matrix),eigvec_v)
#print(s.real.round(3))
#fer_op_s=FermionicOperator(h1=s)
#s_jw=fer_op_s.mapping('jordan_wigner')
#print(s_jw.print_operators())



#sim_dia.append(qt.Qobj(g_matrix))        


#sqr.mapping_with_symmetry_reduction_new(fer_op,si)
