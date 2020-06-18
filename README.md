# Tapering off qubits using the point group symmetries.

This repository contains the code based on qiskit for tapering off qubits using point group symmetries.

It uses qiskit-0.19.4 which can be installed by:

```pip3 install qiskit==0.19.4```

**Python files**
1. r_mat_for_mols.py contains the code for getting the Hamiltonian for a molecule and constructing the R matrices corresponding to the point group symmetries. It also contains the test code which verifies that the R matrices do commute with the Hamiltonian.
2. taper_qubit_rm_funcs.py contains the code that uses the matrices constructed in r_mat_for_mols.py to taper of the qubits. It contains all the helper functions. It main code has two cases for AO and MO basis.
3. symmetries.py contains the code for simultaneously diagonalizing the R matrices.

**Jupyter Notebook**

The demo jupyter notebook `tapering_using_point_group_symmetries.ipynb` provides a step by step process for tapering off qubits using the point group symmetries. The code follows the process/algorithm given in https://arxiv.org/abs/1910.14644