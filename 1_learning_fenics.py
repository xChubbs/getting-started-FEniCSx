######################################################
# author : @alujan
# file : learning_fenics.py
######################################################
# FEniCS self tutorial before working with external
# colleagues, this is a reproduction of usable book's
# codes referenced on the documentation.
######################################################
# @about
# The idea of the file it's to dissect the example
# provided to solve the laplace equation for latter
# implementation of the SAYAS example of use
######################################################

# %% 0. Importing modules
# TODO: I need to solve the non-missing imports highlighted
import ufl
import pyvista
import numpy as np

from mpi4py             import MPI
from dolfinx             import fem, mesh, plot
from petsc4py.PETSc      import ScalarType
from dolfinx.fem.petsc   import LinearProblem

# %% 1. Context definition
# Future idea: import own domains for further use

# - Inner creation of example domain
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(4, 4),
    cell_type=mesh.CellType.triangle,
)

# - Function space definition
V = fem.functionspace(
    msh, ("Lagrange", 2)
)  # Here's used Lagrange polynomials & degree k<=1

# Dirichlet boundary conditions faces
facets = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 2.0),
)

# - Locate of DOF for Dirichlet conditions
dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)

# - Definition of Dirichlet BC type for dofs
bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

# %% 2. Definition of variational problem

# - Definition of functions involved
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# - Retrieve of spacial coordinates for points
x = ufl.SpatialCoordinate(msh)

# - Source (f) and Neumann (g) functions definitions
f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
g = ufl.sin(5 * x[0])

# - Defintion of bilinear form (a) and linear functional (F)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ufl.ds

# %% 3. Solution of variational problem
problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    petsc_options_prefix="demo_poisson_",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
    },
)
uh = problem.solve()

# %% 4. Solution validation
cells, types, x = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.point_data["u"] = uh.x.array.real
grid.set_active_scalars("u")
plotter = pyvista.Plotter()
warped = grid.warp_by_scalar()
plotter.add_mesh(warped, show_edges=True, show_scalar_bar=True, smooth_shading=True)
plotter.show_axes_all()
plotter.show_grid()
plotter.show()
