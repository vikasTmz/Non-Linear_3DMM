import os
import glob
import numpy as np
import random
from PIL import Image, ImageFilter
import math
import operator
import cv2

ROOT = '/mnt/c/Users/vthamizharasan/Desktop/summer_work/geom'
SAVEPATH = ROOT + '/ctx_non_linear_data_texture'


def helpers():

	class VertexUVs(object):
		def __init__(self):
			self.verts = []
			self.uvs = []
			self.colors = []

	def read_ply(obj, filename):
		f = open(filename)
		for line in f:
			line = line.split(' ')
			obj.verts.append('%s %s %s'%(line[0],line[1],line[2].split('\n')[0]))
			if len(line) > 3:
				obj.uvs.append('%s %s'%(line[3],line[4].split('\n')[0]))
			if len(line) > 5:
				obj.uvs.append('%s %s'%(line[3],line[4].split('\n')[0]))


		f.close()

	def write_ply(obj):
		with open('test_uv4.txt', 'w') as the_file:
			for i, vert in enumerate(obj.verts):
				the_file.write('%s %s %s\n'%(vert, obj.colors[i], obj.uvs[i]))

	with_uvs = VertexUVs()
	f = open('test_uv2.txt')
	for line in f:
		line = line.split(' ')
		with_uvs.verts.append('%s %s %s'%(line[0],line[1],line[2].split('\n')[0]))
		with_uvs.uvs.append('%s %s'%(line[3],line[4].split('\n')[0]))
			
	f.close()

	with_colors = VertexUVs()
	f = open('test_uv3.txt')
	for line in f:
		line = line.split(' ')
		with_colors.verts.append('%s %s %s'%(line[0],line[1],line[2].split('\n')[0]))
		with_colors.colors.append('%s %s %s %s'%(line[3],line[4],line[5],line[6]))
		with_colors.uvs.append('%s %s'%(line[7],line[8].split('\n')[0]))
			
	f.close()


	# common = VertexUVs()
	with_uvs.colors = len(with_uvs.verts) * ['0 0 0 255']
	print(len(with_uvs.colors))

	for i,s in enumerate(with_uvs.uvs):
		for j,l in enumerate(with_colors.uvs):
			if s == l:
				with_uvs.colors[i] = with_colors.colors[j]
				break

	write_ply(with_uvs)


def billinearInterpolation(uv, color, texture, filled):
	if uv[0] < 0 or uv[1] < 0:
	    print("Negative uvs")
	    exit()

	height, width, c = texture.shape

	pixelXCoordinate = uv[0] * width - 0.5
	pixelYCoordinate = (1 - uv[1]) * height - 0.5

	if pixelXCoordinate < 0:
	    pixelXCoordinate = width - pixelXCoordinate

	if pixelYCoordinate < 0:
	    pixelYCoordinate = height - pixelYCoordinate

	x = int(math.floor(pixelXCoordinate))
	y = int(math.floor(pixelYCoordinate))

	pX = pixelXCoordinate - x
	pY = pixelYCoordinate - y

	px = [1 - pX,  pX]
	py = [1 - pY, pY]

	for i in range(2):
	    for j in range(2):
	        p = px[i] * py[j]
	        if p != 0:
	            texture[(x + i)%width, (y + j)%height] = color
	            filled[(x + i)%width, (y + j)%height] = True


class ColorWriter(object):
	def __init__(self, ply_filename, obj_filename, texture_filename):
		self.uvs = []
		self.vertex_uv_map = {}
		self.pervertex_colors = []
		self.faces = []
		self.size = 128
		self.texture_map = np.zeros((self.size, self.size,3))
		self.filled = np.zeros((self.size, self.size), dtype=bool)

		try:
			f = open(obj_filename)
			for line in f:
				if line[:3] == "vt ":
					index1 = line.find(" ") + 1
					index2 = line.find(" ", index1 + 1)
					uv = [
							float(line[index1:index2]),
							float(line[index2:])
					]
					self.uvs.append(uv)

				elif line[:2] == "f ":
					def check_append(face_element):
						vid, uvid, _ = face_element.split('/')
						vid = int(vid)
						uvid = int(uvid)
						vid -= 1
						uvid -= 1
						if vid not in self.vertex_uv_map.keys():
							self.vertex_uv_map[vid] = uvid

					index1 = line.find(" ") + 1
					index2 = line.find(" ", index1 + 1)
					index3 = line.find(" ", index2 + 1)
					index4 = line.find(" ", index3 + 1)

					check_append(line[index1:index2])
					check_append(line[index2:index3])
					check_append(line[index3:index4])
					check_append(line[index4:])

					self.faces.append(line)


			self.uvs = np.asarray(self.uvs)
			f.close()

			f = open(ply_filename)
			for line in f:
				line_split = line.split(' ')
				if len(line_split) == 7 and line_split[-1] == '255\n':
					pervertex_color = [
							int(line_split[3]), 
							int(line_split[4]), 
							int(line_split[5])
							]
					self.pervertex_colors.append(pervertex_color)

			f.close()			
			self.pervertex_colors = np.asarray(self.pervertex_colors)
			print(self.uvs.shape, self.pervertex_colors.shape, len(self.vertex_uv_map))
			for vid in self.vertex_uv_map.keys():
				billinearInterpolation(self.uvs[self.vertex_uv_map[vid]],
										self.pervertex_colors[vid],
										self.texture_map,
										self.filled)

			height, width, c = self.texture_map.shape

			while True:
				ufp_rows, ufp_cols = np.where((self.filled == False))
				print(ufp_rows.shape[0], ufp_cols.shape[0])
				if ufp_rows.shape[0] == 0:
					break
				for i,j in zip(ufp_rows, ufp_cols):
					if i==0:
						top = 0
					if i>0:
						top = i-1
					if j==0:
						left = 0
					if j>0:
						left = j-1
					if (i+1) == height:
						bottom = i
					if (i+1) < height:
						bottom = i+1
					if (j+1) == width:
						right = j
					if (j+1) < width:
						right = j+1

					neighbours = [self.texture_map[top,left], self.texture_map[top,j], self.texture_map[top,right],
									self.texture_map[i,left],  self.texture_map[i,right],
									self.texture_map[bottom,left], self.texture_map[bottom,j], self.texture_map[bottom,right]]
					neighbours = np.asarray(neighbours)
					indices = np.where((neighbours[:,0] != 0)&(neighbours[:,1] != 0)&(neighbours[:,2] != 0))[0]
					count = indices.shape[0]
					neighbours_sum = np.sum(neighbours,axis=0)
					self.texture_map[i,j,:] = neighbours_sum // count
					self.filled[i,j] = True

			self.texture_map = self.texture_map.astype(np.uint8)

			self.texture_map = np.rot90(np.fliplr(self.texture_map),1)
			'''sharpening'''
			# self.texture_map = cv2.bilateralFilter(src=self.texture_map, d=5, sigmaColor=35, sigmaSpace=5)
			# kennel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
			# self.texture_map = cv2.filter2D(self.texture_map, -1, kennel)

			self.texture_map = Image.fromarray(self.texture_map)
			'''gaussian'''
			# self.texture_map = self.texture_map.filter(ImageFilter.GaussianBlur(radius = 0.1))


			self.texture_map.save(texture_filename)
		except IOError:
			print("%s file not found." %(fileName))

	def write_obj(self, filename):
		with open(filename, 'w') as the_file:
			for i in range(self.vertices.shape[0]):
				x,y,z = self.vertices[i]
				r,g,b = self.pervertex_color[i]
				the_file.write('v %.3f %.3f %.3f %.3f %.3f %.3f\n' %(x,y,z,r,g,b))

			for f in self.faces:
				the_file.write(f)

def resize_texture(texture_dir, ext):
	size = 1024
	bins = {'bin1':[0,107],'bin2':[107,130],'bin3':[130,140],'bin4':[140,255]}
	textures = glob.glob(os.path.join(ROOT,texture_dir))
	for i in range(0,len(textures)):
		img = Image.open(textures[i])
		img = img.resize((size,size), Image.LANCZOS)
		name = textures[i].split('/')[-1].split(ext)[0]
		width = 100
		out = ''
		# crop_img = np.asarray(img)[390:390+width, 400:400+width*2]
		# # img = Image.fromarray(crop_img)
		# mean = np.mean(crop_img)
		# # mean = np.median(crop_img)
		# for key in bins.keys():
		# 	if mean < bins[key][1] and mean >= bins[key][0]:
		# 		out = key
		img.save(os.path.join(ROOT,'textures_resize','%s.png'%(name)),"PNG")
		# os.system('convert %s %s'%(os.path.join(ROOT,'textures_256x256',name + '.png'),os.path.join(ROOT,'textures_256x256',name + '.jpg')))

def main():

	neutral_mean = ColorWriter(os.path.join(ROOT,'texture_transfer.ply'),
							os.path.join(ROOT,'exp_base.obj'),
							os.path.join(ROOT,'pervertexalbedo2uv_128_filled.png'))


# main()
# texture_dir = 'texture/*.jpg'
texture_dir = 'mapsTransferToIW9UVLayout/*_source_c_iw9.tif'#
# ext = '.jpg'
ext = '.tif'
resize_texture(texture_dir, ext)
