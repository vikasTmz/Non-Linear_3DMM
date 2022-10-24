import numpy as np
import glob
import argparse
import os

DATASETS = {'ctx_albedo':['./data/ctx_ply_face2vertexids.txt','./data/ctx_ply_textcoords.txt']}
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


parser = argparse.ArgumentParser(
            description='Test a Non-Linear 3D Face Morphable Model.'
            )
parser.add_argument('input', type=str, help='')
parser.add_argument('dataset', type=str, help='')

args = parser.parse_args()

ROOT = '/media/vikas/Elements/.temp/.2/' + args.dataset

faces = []
text_coords = []

with open(DATASETS[args.dataset][0], 'r') as the_file:
	Lines = the_file.readlines()
	for line in Lines:
		faces.append(line)

with open(DATASETS[args.dataset][1], 'r') as the_file:
	Lines = the_file.readlines()
	for line in Lines:
		text_coords.append(line)

checkpoint = args.input.split('/')[-1]
os.system('mkdir '+ os.path.join(ROOT, checkpoint))

def write_obj(npzs, geom_key):
	npz = glob.glob(npzs)
	if geom_key == 'gt_geom':
		albedo_key = 'gt_albedo'
	else:
		albedo_key = 'prediction_albedo'

	for output in npz:
		print(output)
		filename = output.split('/')[-1].split('.')[0]
		with open(os.path.join(ROOT, checkpoint, filename + '.ply'), 'w') as the_file:
			obj = np.load(output)
			the_file.write(TEMPLATE)

			for i, vertex  in enumerate(obj[geom_key]):
				x,y,z = vertex
				r,g,b = obj[albedo_key][i] * 255
				uv = text_coords[i]
				the_file.write('%f %f %f %d %d %d 255\n' %(x,y,z,r,g,b))

			for f in faces:
				the_file.write(f)
# try:
write_obj(args.input + '/*_predicted_face.npz', 'prediction_geom')
write_obj(args.input+'/*_gt_face.npz', 'gt_geom')
	# write_obj(args.input+'/*_mean_face.npz')
# except:
	# print("error encountered!")

os.system('rm -rf ' + args.input)
