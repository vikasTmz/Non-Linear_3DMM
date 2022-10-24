import os
import glob
import argparse

import cv2
import numpy as np
import pymeshlab as ml

'''INIT'''
DATASETS = {'ctx':'./data/ctx_obj_texface_properties.txt',
			'ctx_albedo':'./data/ctx_obj_texface_properties.txt',
			'facescape':'./data/facescape_obj_face_properties.txt'}



MLX_TEMPLATE = """
<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Transfer: Vertex Color to Texture">
  <Param tooltip="The texture file to be created" name="textName" value="%s.png" type="RichString" description="Texture file"/>
  <Param tooltip="The texture width" name="textW" value="256" type="RichInt" description="Texture width (px)"/>
  <Param tooltip="The texture height" name="textH" value="256" type="RichInt" description="Texture height (px)"/>
  <Param tooltip="if current mesh has a texture will be overwritten (with provided texture dimension)" name="overwrite" value="false" type="RichBool" description="Overwrite texture"/>
  <Param tooltip="assign the newly created texture" name="assign" value="false" type="RichBool" description="Assign texture"/>
  <Param tooltip="if enabled the unmapped texture space is colored using a pull push filling algorithm, if false is set to black" name="pullpush" value="true" type="RichBool" description="Fill texture"/>
 </filter>
</FilterScript>
"""

'''SYS ARGS'''
parser = argparse.ArgumentParser(
            description='Test a Non-Linear 3D Face Morphable Model.'
            )
parser.add_argument('input', type=str, help='')
parser.add_argument('root', type=str, help='')
parser.add_argument('dataset', type=str, help='')
args = parser.parse_args()


'''DIR SETTINGS'''
ROOT = os.path.join(args.root,args.dataset)
SRTEST = '/media/vikas/Elements/.srtex/pix2pix_VERTCOLOR2LR/AB/test/'

checkpoint = args.input.split('/')[-1]
os.system('mkdir '+ os.path.join(ROOT, checkpoint))



'''LOAD TEMPLATE'''
faces = []
with open(DATASETS[args.dataset], 'r') as the_file:
	Lines = the_file.readlines()
	for line in Lines:
		faces.append(line)


''''''
def write_obj(npzs, geom_key):
	npz = glob.glob(npzs)
	if geom_key == 'gt_geom':
		albedo_key = 'gt_albedo'
	else:
		albedo_key = 'prediction_albedo'

	for output in npz:
		print(output)
		filename = output.split('/')[-1].split('.')[0]
		mlx_filename = os.path.join(ROOT, checkpoint, filename + '.mlx')
		texture_filename = os.path.join(ROOT, checkpoint, filename + '.png')
		obj_filename = os.path.join(ROOT, checkpoint, filename + '.obj')

		with open(obj_filename, 'w') as the_file:
			obj = np.load(output)
			print(obj[geom_key].shape, obj[albedo_key].shape)
			for vert, color in zip(obj[geom_key], obj[albedo_key]):
				x,y,z = vert
				r,g,b = color
				the_file.write('v %f %f %f %f %f %f\n' %(x,y,z,r,g,b))

			for f in faces:
				the_file.write(f)

		if args.dataset == "ctx_albedo":
			with open(mlx_filename, 'w') as the_file:
				the_file.write(MLX_TEMPLATE%(filename))

			ms = ml.MeshSet()
			ms.load_new_mesh(obj_filename)
			ms.load_filter_script(mlx_filename)
			ms.apply_filter_script()

			uv_map = cv2.imread(texture_filename,1)
			cv2.imwrite(os.path.join(SRTEST,filename + '_'  + checkpoint + '.png'),np.concatenate([uv_map, uv_map], 1))

# try:
write_obj(os.path.join(args.input,'*_predicted_face.npz'), 'prediction_geom')
write_obj(os.path.join(args.input,'*_gt_face.npz'), 'gt_geom')

# except:
# 	print("error encountered!")

os.system('rm -rf ' + args.input)
