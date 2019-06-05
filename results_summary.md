# Summary of results:

As per our theoretical work, if we can represent a spatial symmetry using a permutation operator, R, that commutes with the Hamiltonian then we can use the formalism developed in the 'Tapering off qubits' paper to reduce the number of qubits required for the fermionic simulation. So, the code in this repository contains the R matrices for spatial symmetries in different molecules. 

The point group symmetry of a molecule can be described by 5 types of symmetry element:

Symmetry axis (abbreviated Cn): an axis around which a rotation by 360/n results in a molecule indistinguishable from the original.
Plane of symmetry(abbreviated σ): a plane of reflection through which an identical copy of the original molecule is generated. 
Center of symmetry or inversion center: abbreviated i. A molecule has a center of symmetry when, for any atom in the molecule, an identical atom exists diametrically opposite this center an equal distance from it.
Rotation-reflection axis: an axis around which a rotation by 360/n, followed by a reflection in a plane perpendicular to it, leaves the molecule unchanged.
Identity:abbreviated to E, from the German 'Einheit' meaning unity.

Different molecules have different point group symmetries, and they can be classified based on the symmetry elements present in their symmetry group. Based on the the symmetry elements Arthur Moritz Schoenflies came up with a convenient notation for the symmetry group. In our description, we will be using Schoelflies notation.

We now present the spatial symmetries for different molecules. We were able to construct R-matrix for a subset of the spatial symmetries:

1) H2, O2, C2H2, and CO2 (Point Group --> D∞h): We were able to construct R-matrix for following symmetry elements: 
2) NH3 (Point Group --> C3v)
3) H2O (Point Group --> C2v)
4) CH4 (Td)
5) BF3  
