import numpy as np
import os

TEMPLATE = \
"""ply
format ascii 1.0
comment VCGLIB generated
element vertex 20339
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
element face 40616
property list uchar int vertex_indices
end_header
"""

faces = []
with open('./data/ctx_ply_face2vertexids.txt', 'r') as the_file:
	Lines = the_file.readlines()
	for line in Lines:
		faces.append(line)

ROOT = '/mnt/c/Users/vthamizharasan/Desktop/summer_work/geom'
geom = '/mnt/c/Users/vthamizharasan/Desktop/summer_work/geom/ctx_non_linear_data'
albedo = '/mnt/c/Users/vthamizharasan/Desktop/summer_work/geom/ctx_non_linear_data_texture'

target_geom = np.load(geom + '/A_ALFAR_33/target_geom.npz')
target_albedo = np.load(albedo + '/A_ALFAR_0/target_albedo.npz')

target_geom_disp = np.array(target_geom['disp']).astype(np.float32)
mean_geom = np.array(target_geom['mean']).astype(np.float32)   
target_albedo_disp = np.array(target_albedo['disp']).astype(np.float32)
mean_albedo = np.array(target_albedo['mean']).astype(np.float32)

gt_geom = target_geom_disp + mean_geom
gt_albedo = target_albedo_disp + mean_albedo

with open(os.path.join(ROOT, 'texture_transfer1.ply'), 'w') as the_file:
	the_file.write(TEMPLATE)
	for i, vertex  in enumerate(gt_geom):
		x,y,z = vertex
		r,g,b = gt_albedo[i] * 255
		the_file.write('%.3f %.3f %.3f %d %d %d 255\n' %(x,y,z,r,g,b))

	for f in faces:
		the_file.write(f)
