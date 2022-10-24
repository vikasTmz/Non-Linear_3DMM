import os
import glob
import numpy as np
import random
from PIL import Image
import math
import operator

ROOT = '/mnt/c/Users/vthamizharasan/Desktop/summer_work/geom'
# EXPRESSIONS = ['cheekPuffL.obj','cheekSquintL.obj','cheekSquintR.obj','chinRaiser.obj','chinRaiserLowerL.obj','chinRaiserLowerR.obj','dimplerR.obj','dimplerL.obj','eyeCloseL.obj','eyeCloseR.obj','eyeCompressL.obj','eyeCompressR.obj','funnelLowerL.obj','funnelLowerR.obj','funnelUpperL.obj','funnelUpperR.obj','jawOpen.obj','jawOpen_lipCornerPullerL.obj','jawOpen_lipCornerPullerR.obj','jawOpen_lipsTogether_lipCornerPullerL.obj','jawOpen_lipsTogether_lipCornerPullerR.obj','jawOpen_lipsTogether_upperLipRaiserL.obj','jawOpen_lipsTogether_upperLipRaiserR.obj','jawright.obj','lipCornerPullerL.obj','lipCornerPullerR.obj','lipMidLowerDownTweak.obj','lipMidUpperUpTweak.obj','lipOpenMoreLowerL.obj','lipOpenMoreLowerR.obj','lipStretcherL.obj','lipStretcherR.obj','mouthSwingL.obj','mouthSwingR.obj','noseWrinklerR.obj','upperLipRaiserR.obj','upperLipRaiserL.obj']
EXPRESSIONS = ['exp_base.obj'] #'eyeRaiseL.obj','eyeRaiseR.obj','lipsTogether.obj','stickyLipsR4.obj','eyeRightDL.obj','eyeClose_eyeLeftL.obj','lipCloseMoreUpR.obj','uprLipDepressR.obj','lipPressR.obj','jawOpen_lowerLipDepressorL.obj','lipsTogether_lowerLipDepressorR.obj','eyeDownR.obj','eyeClose_eyeCompressL.obj','eyeCloseL.obj','lowerLipDepressorL.obj','lipCornerPuller_noseWrinklerL.obj','jawOpen_lowerLipDepressorR.obj','eyeDownL.obj','eyeClose_eyeCompressR.obj','chinRaiserLowerR.obj','cheekSuckL.obj','neckTenseR.obj','jawOpen_lipsTogether_lipCornerPullerR.obj','chinRaiserLowerL.obj','cheekSquintR.obj','lipTightenerR.obj','puckerLowerL.obj','lipSuckUpperR.obj','squintL.obj','puckerUpperL.obj','noseWrinklerL.obj','puckerUpperR.obj','lipCornerPullerL.obj','noseWrinklerR.obj','dimplerR.obj','dimplerL.obj','chinRaiser.obj','lipCornerPullerR.obj','jawback.obj','jawright.obj','jawleft.obj','mouthSwingR.obj','mouthSwingL.obj','jawOpen_lipsTogether.obj','jawOpen.obj']
DIAGONAL_SIZE = 1
CENTER_COORD = 0.5

def get_pc_diag(pc):
	 xwidth = np.amax(pc[:,0]) - np.amin(pc[:,0])
	 ywidth = np.amax(pc[:,1]) - np.amin(pc[:,1])
	 zwidth = np.amax(pc[:,2]) - np.amin(pc[:,2])
	 diagonal_len = np.sqrt(xwidth**2 + zwidth**2 + ywidth**2)
	 return diagonal_len

def normalize_pc(pc, size=1):
	 ''' Normalize so diagonal of tight bounding box is 1 '''
	 diagonal_len_sqr = get_pc_diag(pc)**2
	 norm_pc = pc * np.sqrt(size / diagonal_len_sqr)
	 return norm_pc

def billinearInterpolation(uv, texture):
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

	red = 0
	green = 0
	blue = 0
	alpha = 0

	for i in range(2):
	    for j in range(2):
	        p = px[i] * py[j]
	        if p != 0:
	            rgb = texture[(x + i)%width, (y + j)%height]
	            alpha += rgb[3] * p
	            red += rgb[0] * p
	            green += rgb[1] * p
	            blue += rgb[2] * p

	return [red/255,green/255,blue/255]

# from https://stackoverflow.com/a/32558749/6386471
def find_best_match(string, matchings):
	scores = {}
	def levenshteinDistance(s1, s2):
	    if len(s1) > len(s2):
	        s1, s2 = s2, s1

	    distances = range(len(s1) + 1)
	    for i2, c2 in enumerate(s2):
	        distances_ = [i2+1]
	        for i1, c1 in enumerate(s1):
	            if c1 == c2:
	                distances_.append(distances[i1])
	            else:
	                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
	        distances = distances_
	    return distances[-1]

	for m in matchings:
		scores[m] = 1 - levenshteinDistance(string,m)

	return max(scores.items(), key=operator.itemgetter(1))[0]


class ObjLoader(object):
	def __init__(self, obj_filename, texture_filename=None):
		self.vertices = []
		self.uvs = []
		self.vertex_uv_map = {}
		self.pervertex_color = []
		self.faces = []
		self.texture_map = None
		if texture_filename is not None:
			self.texture_map = np.array(Image.open(texture_filename))
			self.texture_map = np.rot90(np.fliplr(self.texture_map),1)

		try:
			f = open(obj_filename)
			for line in f:
				if line[:2] == "v ":
					index1 = line.find(" ") + 1
					index2 = line.find(" ", index1 + 1)
					index3 = line.find(" ", index2 + 1)
					vertex = [
							round(float(line[index1:index2]),6), 
							round(float(line[index2:index3]),6),
							round(float(line[index3:-1]),6)
							]
					self.vertices.append(vertex)

				elif line[:3] == "vt ":
					index1 = line.find(" ") + 1
					index2 = line.find(" ", index1 + 1)
					uv = [
							float(line[index1:index2]),
							float(line[index2:])
					]
					self.uvs.append(uv)

				elif line[:2] == "f ":
					def check_append(face_element):
						face_element = face_element.split('/')
						vid = face_element[0]
						uvid = face_element[1]
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


			self.vertices = np.asarray(self.vertices)
			self.pervertex_color = np.zeros(self.vertices.shape)

			if self.texture_map is not None:
				self.uvs = np.asarray(self.uvs)
				for vid in self.vertex_uv_map.keys():
					self.pervertex_color[vid] = billinearInterpolation(self.uvs[self.vertex_uv_map[vid]], self.texture_map)

			self.vertices = normalize_pc(self.vertices, DIAGONAL_SIZE)
			self.vertices_h = np.ones((self.vertices.shape[0],4))
			self.vertices_h[:,:3] = self.vertices
			T = np.array([[1,0,0,CENTER_COORD], [0,1,0,CENTER_COORD], [0,0,1,CENTER_COORD], [0,0,0,1]])
			self.vertices = np.dot(T,self.vertices_h.T).T[:,:3]
			f.close()
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

def compute_mean(neutral_component):
	return np.mean(neutral_component, axis=0)\

def measure_exp_deviation():
	neutral_mean = ObjLoader(os.path.join(ROOT,'exp_base.obj'))
	expressions = glob.glob(os.path.join(ROOT,'exp/*.obj'))
	distance = {}
	for i, exp in enumerate(expressions):
		expression_mean = ObjLoader(exp)
		name = exp.split('/')[-1].split('.')[0]
		distance[name] = np.round(np.linalg.norm((expression_mean.vertices - neutral_mean.vertices), ord=1), 4)

	distance = {k: v for k, v in sorted(distance.items(), key=lambda item: item[1])}
	print(distance)

def create_geometry_model_dataset(identities):
	neutral_mean = ObjLoader(os.path.join(ROOT,'exp_base.obj'))

	for i, exp in enumerate(EXPRESSIONS):
		# try:
		print(exp, i)
		filename = os.path.join(ROOT,'exp',exp)
		expression_mean = ObjLoader(filename)

		exp_disp = expression_mean.vertices - neutral_mean.vertices

		blendweight = np.zeros((len(EXPRESSIONS),1))
		blendweight[i] = 1
		blendweight = {'blendweight':blendweight}

		for idx in identities:
			# try:

			obj = ObjLoader(idx)
			neutral_disp_geom = np.round(obj.vertices - neutral_mean.vertices, 5)
			target_disp_geom = np.round(neutral_disp_geom + exp_disp, 5)
			# save data
			name = idx.split('/')[-1].split('.')[0]
			os.system('mkdir %s/%s_%d' %(SAVEPATH,name,i))
			neutral_geom = {'disp':neutral_disp_geom}
			target_geom = {'disp':target_disp_geom,'mean':neutral_mean.vertices}
			target_albedo = {'disp':np.zeros((1,1)),'mean':np.zeros((1,1))}

			path = os.path.join(SAVEPATH,name + '_' + str(i))
			np.savez(os.path.join(path,'neutral_geom.npz'), **neutral_geom)
			np.savez(os.path.join(path,'target_geom.npz'), **target_geom)
			np.savez(os.path.join(path,'target_albedo.npz'), **target_albedo)
			np.savez(os.path.join(path,'blendweight.npz'), **blendweight)
				# except:
					# print("object missing = %s_%d" %(idx,i))

		# except:
			# print("error in processing expression")

def create_texture_model_dataset(identities):
	neutral_mean = ObjLoader(os.path.join(ROOT,'exp_base.obj'), os.path.join(ROOT,'mean_neutral_texture.tif'))
	# neutral_mean.write_obj(os.path.join(ROOT,'exp_base_color.obj'))
	'''Texture Start'''
	textures = glob.glob(os.path.join(ROOT,'mapsTransferToIW9UVLayout/*_c_iw9.tif'))

	neutral_textures = []
	if not os.path.isfile(os.path.join(ROOT,'mean_neutral_texture.tif')):
		for i in range(len(textures)):
			text = textures[i]
			text_tif = np.array(Image.open(text))
			neutral_textures.append(text_tif)
		neutral_textures = np.array(neutral_textures)
		mean_neutral_texture = compute_mean(neutral_textures)

		mean_neutral_texture = mean_neutral_texture.astype(np.uint8)
		mean_neutral_texture = Image.fromarray(mean_neutral_texture)
		mean_neutral_texture.save(os.path.join(ROOT,'mean_neutral_texture.tif'))
	else:
		mean_neutral_texture = Image.open(os.path.join(ROOT,'mean_neutral_texture.tif'))
		mean_neutral_texture = np.array(mean_neutral_texture)
	'''Texture End'''

	blendweight = np.zeros((len(EXPRESSIONS),1))
	blendweight[0] = 1
	blendweight = {'blendweight':blendweight}

	for i in range(len(textures)): # 
		texture_filename = textures[i].split('/')[-1]
		matching_string = '_'.join(texture_filename.split('_')[:2])
		obj_filename = find_best_match(matching_string, identities)

		if texture_filename == 'B_KARGORGIS_FACIALHAIR_source_c_iw9.tif':
			obj_filename = os.path.join(ROOT,'id','B_KARGORGIS_FACIALHAIR.obj')
		elif texture_filename == 'A_ZEDRA_NEUTRAL_EYESOPEN-12_source_c_iw9.tif':
			obj_filename = os.path.join(ROOT,'id','A_ZEDRA_NEUTRAL_EYESOPEN-12.obj')
		elif texture_filename == 'P_Grier_source_c_iw9.tif':
			obj_filename = os.path.join(ROOT,'id','P_Grier.obj')

		idx = obj_filename
		print(texture_filename, obj_filename)
		try:

			obj = ObjLoader(idx, textures[i])
			# obj.write_obj(os.path.join(ROOT,obj_filename))
			neutral_disp_geom = np.round(obj.vertices - neutral_mean.vertices,8)
			neutral_disp_albedo = np.round(obj.pervertex_color - neutral_mean.pervertex_color,8)
			target_disp_geom = neutral_disp_geom
			target_disp_albedo = neutral_disp_albedo
			# save data
			name = idx.split('/')[-1].split('.')[0]
			os.system('mkdir %s/%s_%d' %(SAVEPATH,name,0))
			neutral_geom = {'disp':neutral_disp_geom}
			target_geom = {'disp':target_disp_geom,'mean':neutral_mean.vertices}
			target_albedo = {'disp':target_disp_albedo,'mean':neutral_mean.pervertex_color}

			path = os.path.join(SAVEPATH,name + '_0')
			np.savez(os.path.join(path,'neutral_geom.npz'), **neutral_geom)
			np.savez(os.path.join(path,'target_geom.npz'), **target_geom)
			np.savez(os.path.join(path,'target_albedo.npz'), **target_albedo)
			np.savez(os.path.join(path,'blendweight.npz'), **blendweight)
		except:
			print("object missing = %s_%d" %(idx,i))

def create_geometry_expression_combination_dataset(identities):
	SAVEPATH = ROOT + '/ctx_non_linear_geom_expression_combination'

	COMBINATIONS = {
	# 'neutral':['exp_base'],
	'surprise': ['jawOpen','eyeClose_eyeCompressL','eyeClose_eyeCompressR','procerusLowerL','procerusLowerR','jawOpen_neckTenseL','jawOpen_neckTenseR'],
	'smile': ['sharpCornerPullerL','sharpCornerPullerR','noseFlareL','noseFlareR','lidTightR','lidTightL'],
	'right_wink_cheek': ['lipCornerPullerR','eyeCompressR'],
	'left_wink_cheek': ['lipCornerPullerL','eyeCompressL'],
	'sad': ['dimplerL','dimplerR','chinRaiser_lipsTogether','lipStretcher_neckTenseR','lipStretcher_neckTenseL','eyeLeftR','eyeRightL'],
	'angry': ['cheekSquintL','cheekSquintR','cheekSuckL','cheekSuckR','lidUpTweakL','lidUpTweakR'],
	'worried': ['chinRaiser','lipStretcherR','lipStretcherL','procerusUpL','procerusUpR','nostrilDepressor'],
	'mouthswingL':['mouthSwingL','neckTenseL','squintL'],
	'mouthswingR':['mouthSwingR','neckTenseR','squintR','eyeCloseR'],
	'funkyR':['funnelLowerR','funnelUpperL','eyeDownL','furrowL','eyeRightR'],
	'funkyL':['funnelLowerL','funnelUpperR','eyeDownR','furrowR','eyeLeftL'],
	'cheekpuffR':['jawOpen_lipCornerPullerL','cheekPuffR','noseWrinklerR','eyeClose_eyeRightR'],
	'cheekpuffL':['jawOpen_lipCornerPullerR','cheekPuffL','noseWrinklerL','eyeClose_eyeLeftL']
	}

	neutral_mean = ObjLoader(os.path.join(ROOT,'exp_base.obj'))

	for i, key in enumerate(COMBINATIONS.keys()):
		# try:
		print(key, i)
		combined_exp_disp = np.zeros((neutral_mean.vertices.shape))
		for exp in COMBINATIONS[key]:
			filename = os.path.join(ROOT,'exp',exp + '.obj')
			expression_mean = ObjLoader(filename)
			exp_disp = expression_mean.vertices - neutral_mean.vertices
			combined_exp_disp += exp_disp

		blendweight = np.zeros((len(COMBINATIONS.keys()),1))
		blendweight[i] = 1
		blendweight = {'blendweight':blendweight}

		for idx in identities:
			# try:

			obj = ObjLoader(idx)
			neutral_disp_geom = np.round(obj.vertices - neutral_mean.vertices, 5)
			target_disp_geom = np.round(neutral_disp_geom + combined_exp_disp, 5)
			
			name = idx.split('/')[-1].split('.')[0]
			obj.vertices += combined_exp_disp
			obj.write_obj('%s_%d.obj'%(name,i))
			break

			# # save data
			# name = idx.split('/')[-1].split('.')[0]
			# os.system('mkdir %s/%s_%d' %(SAVEPATH,name,i))
			# neutral_geom = {'disp':neutral_disp_geom}
			# target_geom = {'disp':target_disp_geom,'mean':neutral_mean.vertices}
			# target_albedo = {'disp':np.zeros((1,1)),'mean':np.zeros((1,1))}

			# path = os.path.join(SAVEPATH,name + '_' + str(i))
			# np.savez(os.path.join(path,'neutral_geom.npz'), **neutral_geom)
			# np.savez(os.path.join(path,'target_geom.npz'), **target_geom)
			# np.savez(os.path.join(path,'target_albedo.npz'), **target_albedo)
			# np.savez(os.path.join(path,'blendweight.npz'), **blendweight)

				# except:
					# print("object missing = %s_%d" %(idx,i))
		break
		# except:
			# print("error in processing expression")

def main():
	identities_capture = glob.glob(os.path.join(ROOT,'id/*.obj'))
	identities_multilinear = glob.glob(os.path.join(ROOT,'obj/*.obj'))
	random.shuffle(identities_capture)


	# measure_exp_deviation()
	# create_geometry_model_dataset(identities_capture + identities_multilinear[0:700])
	# create_texture_model_dataset(identities_capture)
	create_geometry_expression_combination_dataset(identities_capture)

main()
