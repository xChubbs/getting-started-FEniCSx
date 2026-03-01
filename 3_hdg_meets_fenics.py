######################################################
# author : @alujan
# file : hdg_meets_fenics.py
######################################################
# FEniCS self tutorial before working with external
# colleagues, this is a reproduction of usable book's
# codes referenced on the documentation.
######################################################
# @about
# The idea of this file it's to dissect the example
# provided by FEniCS documentation and deeply
# understand the implementation of the HDG method.
######################################################

# %% 0. Importing modules
import ufl
import numpy as np

from mpi4py     import MPI
from petsc4py   import PETSc

import dolfinx

from dolfinx            import fem, mesh, io
from dolfinx.cpp.mesh   import cell_num_entities
from dolfinx.fem        import extract_function_spaces
from dolfinx.fem.petsc  import (
                               apply_lifting,
                               assemble_matrix,
                               assemble_vector,
                               assign,
                               create_vector,
                               set_bc,
                               )

# %% 1. Definition of additional functions neededl
# Parallel norm computation
def norm_L2(v: ufl.core.expr.Expr, measure: ufl.Measure = ufl.dx) -> np.inexact:
    """
    Convenience function to compute the L2 norm of a UFL expression.
    """

    compiled_form = fem.form(ufl.inner(v, v) * measure)
    comm = compiled_form.mesh.comm
    return np.sqrt(comm.allreduce(fem.assemble_scalar(compiled_form), op=MPI.SUM))


# Compute of cell vs. facets
def compute_cell_boundary_facets(msh : dolfinx.mesh.Mesh) -> np.inexact:
    """
    Compute the integration entities for integrals around the
    boundaries of all cells in msh.

    Parameters:
        msh: The mesh.

    Returns:
        Facets to integrate over, identified by `(cell, local facet
        index)` pairs.
    """
    tdim = msh.topology.dim
    fdim = tdim - 1
    n_f = cell_num_entities(msh.topology.cell_type, fdim)
    n_c = msh.topology.index_map(tdim).size_local
    return np.vstack(
        (np.repeat(np.arange(n_c), n_f), np.tile(np.arange(n_f), n_c))
    ).T.flatten()


# Exact solution of the test problem
def u_e(x):
    """Exact solution."""
    u_e = 1
    for i in range(tdim):
        u_e *= ufl.sin(ufl.pi * x[i])
    return u_e


# %% 2. Context definition of the problem
comm = MPI.COMM_WORLD       # Parallel backend
rank = comm.rank            # Number of processors
dtype = PETSc.ScalarType    # Scalar type

# %% 3. Loading of mesh of interest
# Loading a gmsh mesh, it's important to remember that the mesh loads
# the identities:
# - msh : Mesh identity of dolfinx.
# - markers : Cell markers from gmsh.
# - facets : Facet markers from gmsh.
msh, markers, facets = io.gmsh.read_from_msh(
    "./include/gmsh/cubeD0.msh", comm)[:3]

# Dimentionality of the interpolants
tdim = msh.topology.dim     # volume variables dimention
fdim = tdim - 1             # surface variables dimention

# Definition of facets over generated mesh:
# Here we define the identities of the faces to be able to modify
# and create function spaces over the faces.
msh.topology.create_entities(fdim)

facet_imap = msh.topology.index_map(fdim)                   # Faces mapping
num_facets = facet_imap.size_local + facet_imap.num_ghosts  # #Faces + #Non-tagged faces
facets     = np.arange(num_facets, dtype=np.int32)          # Sort of faces (ascending)

# Generation of face mesh (skeleton)
# Here it's important to notice that we recieve a `facet_mesh_emap` this is the mapping
# from the msh faces onto the face_mesh entities.
# - Due to the use of dual spaces mesh vs. skeleton it's needed to define a mesh
#   containing just the faces of the original mesh
facet_mesh, facet_mesh_emap = mesh.create_submesh(msh, fdim, facets)[:2]

entity_maps = [facet_mesh_emap]     # Listing of facet mesh to mesh

# %% 4. Definition of function spaces
k = 1  # Polynomial order

# Definition of function spaces
# It's important to remark that here we define DISCONTINUOUS polynomial spaces,
# just by this the Dolphinx framework knows we're working on a DG method.
V    = fem.functionspace(       msh, ("Discontinuous Lagrange", k))
Vbar = fem.functionspace(facet_mesh, ("Discontinuous Lagrange", k))

# Coupling of function spaces V & Vbar (Volume & Skeleton)
# By this coupling we're now remarking that the approximation in the space V is
# related to the approximation in the space Vbar through the facet mesh.
W = ufl.MixedFunctionSpace(V, Vbar)     # Definition of space W = V \times \bar{V}

# Definition of approximation and test functions
u, ubar = ufl.TrialFunctions(W)
v, vbar = ufl.TestFunctions(W)

# %% 5. Definition of measurements
dx_c = ufl.Measure("dx", domain=msh)  # Cell measure

# We're now defining a measure to integrate around the boundary
# of each cell
cell_boundary_facets = compute_cell_boundary_facets(msh)
cell_boundaries = 1                                         # Tag to denote \partial K (msh)

# Definition of measure over the boundary of each cell
ds_c = ufl.Measure(
    "ds", subdomain_data=[(cell_boundaries, cell_boundary_facets)], domain=msh
)

# Create a cell integral measure over the facet mesh
dx_f = ufl.Measure("dx", domain=facet_mesh)

# %% 6. Variational formulation
# Definition of mesh variables
h = ufl.CellDiameter(msh)   # Cell diameter
n = ufl.FacetNormal(msh)    # Facet normals

x = ufl.SpatialCoordinate(msh)              # Definition spatial coordinates
c = 1.0 + 0.1 * (ufl.sin(ufl.pi * x[0])     # Definition of coefficient c
               * ufl.sin(ufl.pi * x[1]))

gamma = 1                                   # Scaled penalty parameter

# Definition of bilinear form a
a = (
    ufl.inner(c * ufl.grad(u), ufl.grad(v)) * dx_c
    - ufl.inner(c * (u - ubar), ufl.dot(ufl.grad(v), n)) * ds_c(cell_boundaries)
    - ufl.inner(ufl.dot(ufl.grad(u), n), c * (v - vbar)) * ds_c(cell_boundaries)
    + gamma * ufl.inner(c * (u - ubar), v - vbar) * ds_c(cell_boundaries)
)

# Source term
f = -ufl.div(c * ufl.grad(u_e(x)))

# Linear form L definition (RHS)
L = ufl.inner(f, v) * dx_c
L += ufl.inner(fem.Constant(facet_mesh, dtype(0.0)), vbar) * dx_f

# Compile of forms for the block system
a_blocked = dolfinx.fem.form(ufl.extract_blocks(a),     # a Bilinear block
                            entity_maps=entity_maps)
L_blocked = dolfinx.fem.form(ufl.extract_blocks(L))     # L Linear Block

# %% 7. Addition of boundary conditions
# Addition of exterior facet mesh conditions
msh_boundary_facets = mesh.exterior_facet_indices(msh.topology)

# Extraction of the imposed conditions mappings
facet_mesh_boundary_facets = facet_mesh_emap.sub_topology_to_topology(
                                msh_boundary_facets, inverse=True
                            )

# Recovery of DOF's associated with Dirichlet conditions
facet_mesh.topology.create_connectivity(fdim, fdim)
dofs = fem.locate_dofs_topological(Vbar, fdim, facet_mesh_boundary_facets)
bc = fem.dirichletbc(dtype(0.0), dofs, Vbar)

# %% 8. Assembly of the final system
# Assemble of bilinear with boundary conditions
A = assemble_matrix(a_blocked, bcs=[bc])    # Here the matrix A is formed by the blocks of a
A.assemble()                                # And we compile the matrix.

# Assemble of RHS with existing conditions
b = assemble_vector(L_blocked)                  # RHS vector assembled from L_blocked
bcs1 = fem.bcs_by_block(
    extract_function_spaces(a_blocked, 1),      # Boundary conditions assembled by blocks
    [bc])

# Lifting of solution into the \Gamma Dirichlet
apply_lifting(b, a_blocked, bcs=bcs1)           # Lifting of block matrixes

# Update of RHS according to the lifting
b.ghostUpdate(addv=PETSc.InsertMode.ADD,        # Update of values after lifting
              mode=PETSc.ScatterMode.REVERSE)

# Boundary conditions extracted after lifting \Gamma_{BC} -> \Gamma_0
bcs0 = fem.bcs_by_block(
    extract_function_spaces(L_blocked), [bc])
set_bc(b, bcs0)

# %% 9. Construction of Solver
ksp = PETSc.KSP().create(msh.comm)      # Mapping into Whole system matrix

ksp.setOperators(A)                     # Operator Associated matrix

# Additional solver arguments
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

# %% 10. Computing of solution
# - Casing is used in instances the solution it's not obtained
try:
    x = create_vector([V, Vbar])    # Definition of solution vector
    ksp.solve(b, x)                 # Solution of system by KSP solver
    ksp.destroy()                   # Memmory reduction of Solver

    # Update of values obtained
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                  mode=PETSc.ScatterMode.FORWARD)

    # Memmory reduction of RHS
    b.destroy()

# Exception in case of the solver not existing method
except PETSc.Error as e:
    if e.ierr == 92:
        print("The required PETSc solver/preconditioner is not available. Exiting.")
        print(e)
        exit(0)
    else:
        raise e

# %% 11. Creation of solution approximation function
# u and \hat{u} definition for each space
u, ubar = fem.Function(V, name="u"), fem.Function(Vbar, name="ubar")

assign(x, [u, ubar])    # Assignation of solution into variables
x.destroy()             # Memmory reduction of solution vector

# %% 12.  Computation of errors
x = ufl.SpatialCoordinate(msh)              # Volume coordinates
e_u = norm_L2(u - u_e(x))                   # Norm computation error: Volume

x_bar = ufl.SpatialCoordinate(facet_mesh)   # Face coordinates
e_ubar = norm_L2(ubar - u_e(x_bar))         # Norm computarion error: Faces

# Printing of errors
PETSc.Sys.Print(f"e_u = {e_u:.5e}")
PETSc.Sys.Print(f"e_ubar = {e_ubar:.5e}")
