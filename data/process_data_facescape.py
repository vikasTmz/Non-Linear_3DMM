import os
import glob
import numpy as np

ROOT = "/media/vikas/Elements/facescape"
SAVEPATH = "/media/vikas/Elements/facescape_nonlinear_data"
EXPRESSIONS = ['1_neutral.obj','2_smile.obj','3_mouth_stretch.obj','4_anger.obj',
			'5_jaw_left.obj','6_jaw_right.obj','7_jaw_forward.obj','8_mouth_left.obj',
			'9_mouth_right.obj','10_dimpler.obj','11_chin_raiser.obj','12_lip_puckerer.obj',
			'13_lip_funneler.obj','14_sadness.obj','15_lip_roll.obj','16_grin.obj',
			'17_cheek_blowing.obj','18_eye_closed.obj','19_brow_raiser.obj','20_brow_lower.obj']

def get_pc_diag(pc):
	 xwidth = np.amax(pc[:,0]) - np.amin(pc[:,0])
	 ywidth = np.amax(pc[:,1]) - np.amin(pc[:,1])
	 zwidth = np.amax(pc[:,2]) - np.amin(pc[:,2])
	 diagonal_len = np.sqrt(xwidth**2 + zwidth**2 + ywidth**2)
	 return diagonal_len

def normalize_pc(pc):
	 ''' Normalize so diagonal of tight bounding box is 1 '''
	 diagonal_len_sqr = get_pc_diag(pc)**2
	 norm_pc = pc * np.sqrt(1.0 / diagonal_len_sqr)
	 return norm_pc

class ObjLoader(object):
	def __init__(self, fileName):
		self.vertices = []

		try:
			f = open(fileName)
			for line in f:
				if line[:2] == "v ":
					index1 = line.find(" ") + 1
					index2 = line.find(" ", index1 + 1)
					index3 = line.find(" ", index2 + 1)

					vertex = [round(float(line[index1:index2]),6), round(float(line[index2:index3]),6), round(float(line[index3:-1]),6)]
					self.vertices.append(vertex)

			self.vertices = np.asarray(self.vertices)
			self.vertices = normalize_pc(self.vertices)
			self.vertices_h = np.ones((self.vertices.shape[0],4))
			self.vertices_h[:,:3] = self.vertices
			T = np.array([[1,0,0,0.5], [0,1,0,0.5], [0,0,1,0.5], [0,0,0,1]])
			self.vertices = np.dot(T,self.vertices_h.T).T[:,:3]
			f.close()
		except IOError:
			print(".obj file not found.")	

def compute_mean_geom(neutral_meshes):
	return np.round(np.mean(neutral_meshes, axis=0),6)


def main():
	neutral_meshes = []
	for idx in range(1,311):
		try:
			filename = os.path.join(ROOT,str(idx),'models_reg','1_neutral.obj')
			obj = ObjLoader(filename)
			neutral_meshes.append(obj.vertices)
		except:
			print("not included in mean = ",idx)

	neutral_meshes = np.asarray(neutral_meshes)
	R_geom = compute_mean_geom(neutral_meshes)

	for idx in range(530,541):
		neutral_disp_geom = []
		idx = str(idx)
		for exp in range(1,21):
			try:
				filename = os.path.join(ROOT,idx,'models_reg',EXPRESSIONS[exp-1])
				blendweight = np.zeros((20,1))
				blendweight[exp-1] = 1
				obj = ObjLoader(filename)
				disp_geom = obj.vertices - R_geom
				if exp == 1:
					neutral_disp_geom = {'disp':disp_geom}
				# save data
				os.system('mkdir %s/%s_%d' %(SAVEPATH,idx,exp))
				target_geom = {'disp':disp_geom,'mean':R_geom}
				target_albedo = {'disp':np.zeros((1,3)),'mean':np.zeros((1,3))}
				blendweight = {'blendweight':blendweight}
				np.savez(os.path.join(SAVEPATH,idx+'_'+str(exp),'neutral_geom.npz'), **neutral_disp_geom)
				np.savez(os.path.join(SAVEPATH,idx+'_'+str(exp),'target_geom.npz'), **target_geom)
				np.savez(os.path.join(SAVEPATH,idx+'_'+str(exp),'target_albedo.npz'), **target_albedo)
				np.savez(os.path.join(SAVEPATH,idx+'_'+str(exp),'blendweight.npz'), **blendweight)
			except:
				print("object missing = %s_%d" %(idx,exp))

main()
