# coding: utf-8
""" demo on static solving using JAC (experimental) """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

# pyEIT 2D algorithms modules
from pyeit.mesh import create, set_perm_list, set_perm
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines
import pyeit.eit.jac as jac
#from stl import mesh
import trimesh

""" 1. setup """
n_el = 16

mesh_obj, el_pos = create(n_el, h0=0.1)

mesh_obj = {'node': [], 'element':[], 'perm':[]}


##############################
# Using an existing stl file:
#your_mesh = mesh.Mesh.from_file('/media/win/NewLung/GeometricModels/3D_Digitise/ComFiles/thorax_a05.stl')

#VERTICE_COUNT = 100
#data = numpy.zeros(VERTICE_COUNT, dtype=mesh.Mesh.dtype)
#your_mesh = mesh.Mesh(data, remove_empty_areas=False)
#your_mesh.normals
#your_mesh.v0, your_mesh.v1, your_mesh.v2

#axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))
#scale = your_mesh.points.flatten(-1)
#axes.auto_scale_xyz(scale, scale, scale)
#pyplot.show()

#mesh_obj, el_pos = create(n_el, h0=0.1)
##############################

mesh = trimesh.load('thorax_a05_coarse.stl', process=False)

mesh_obj['node'] = mesh.vertices
mesh_obj['element'] = mesh.faces


# test function for altering the permittivity in mesh

#anomaly = {"num": [0], "perm":[]}
#anomaly = {'num':[]}
#facetcount = 0
#for facet in mesh.faces:
#   facetcount = facetcount+1
#   if(facetcount>1312):
#       anomaly['num'].append(facetcount)

#anomaly = [{'num': 1312, 'perm': 10},
#          {'num': 1313, 'perm': 10}]
# background changed to values other than 1.0 requires more iterations
#mesh_new = set_perm_list(mesh_obj, anomaly=anomaly, background=2.)

#anomaly = [{'x': 0.4, 'y': 0.4, 'd': 2.2, 'perm': 10},
#          {'x': -0.4, 'y': -0.4, 'd': 2.2, 'perm': 0.1}]
#mesh_new = set_perm(mesh_obj, anomaly=anomaly, background=2.)

# extract node, element, perm
pts = mesh_obj['node']
tri = mesh_obj['element']
perm = mesh_new['perm']

# show
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.tripcolor(pts[:, 0], pts[:, 1], tri,
                  np.real(perm), shading='flat', cmap=plt.cm.viridis)
fig.colorbar(im)
ax.axis('equal')
ax.set_title(r'$\Delta$ Conductivities')
plt.show()

""" 2. calculate simulated data """
el_dist, step = 1, 1
ex_mat = eit_scan_lines(n_el, el_dist)

fwd = Forward(mesh_obj, el_pos)
f1 = fwd.solve_eit(ex_mat, step, perm=mesh_new['perm'], parser='std')


# draw electrodes and show equipotential lines
ax1.plot(x[el_pos], y[el_pos], 'ro')
for i, e in enumerate(el_pos):
    ax1.text(x[e], y[e], str(i+1), size=12)
ax1.set_title('equi-potential lines')

# clean up
ax1.set_aspect('equal')
ax1.set_ylim([-1.2, 1.2])
ax1.set_xlim([-1.2, 1.2])
fig.set_size_inches(6, 6)
# fig.savefig('demo_bp.png', dpi=96)
plt.show()


""" 3. solve_eit using gaussian-newton (with regularization) """
# number of stimulation lines/patterns
eit = jac.JAC(mesh_obj, el_pos, ex_mat, step, perm=1.0, parser='std')
eit.setup(p=0.25, lamb=1.0, method='lm')
# lamb = lamb * lamb_decay
ds = eit.gn(f1.v, lamb_decay=0.1, lamb_min=1e-5, maxiter=20, verbose=True)


# plot
#-------------------
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, np.real(ds),
                  shading='flat', alpha=1.0, cmap=plt.cm.viridis)
fig.colorbar(im)
ax.axis('equal')
ax.set_title('Conductivities Reconstructed')
# fig.savefig('../figs/demo_static.png', dpi=96)
plt.show()
