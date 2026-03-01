######################################################
# author : @alujan
# file : loading_gmsh_msh.py
######################################################
# FEniCS self tutorial before working with external
# colleagues, this is a reproduction of usable book's
# codes referenced on the documentation.
######################################################
# @about
# The idea of the file it's to load a mesh from a
# gmsh file
######################################################

# %% 0. Importing modules
import ufl
import gmsh
import numpy as np

from mpi4py     import MPI
from dolfinx    import fem, io

# %% 1. Loading a gmsh mesh
domain, markers, facets = io.gmsh.read_from_msh("./include/gmsh/cubeD0.msh", MPI.COMM_WORLD)[:3]

# Definition of differentials
dx = ufl.Measure("dx", domain=domain, subdomain_data=markers)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facets)

# Characteristics of domain
print(f'shape: {domain.topology.dim}')

# %% 2. Testing of loaded structure
# Volume test -> we should recieve the expected Volume
volume = fem.assemble_scalar(fem.form(1.0 * dx))
print(f"Expected volume = {1**3:.3f}   Computed volume = {volume:.3f}")

# Areas test -> we should recieve the expected area
area = fem.assemble_scalar(fem.form(1.0 * ds(50)))
print(f"Expected area = {1**2 * 6:.3f}   Computed area = {area:.3f}")
