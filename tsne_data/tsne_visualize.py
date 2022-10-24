import json
import cv2
import numpy as np
import os
import glob
import operator

import matplotlib.pyplot as plt
import itertools
from scipy.spatial import ConvexHull

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

def search_dict(string):
	known_images = {}
	if string in known_images.keys():
		return known_images[string]
	else:
		return False

def old():
	img_scale = 15
	gt_image_scale = 0.3
	scatter_distance = 10

	coords = np.array(list(tsne_dict.values()))

	image_width = int(np.max(coords[:,0]) - np.min(coords[:,0]))
	image_width *= img_scale
	img_cx = image_width // 2

	image_height = int(np.max(coords[:,1]) - np.min(coords[:,1]))
	image_height *= img_scale
	img_cy = image_height // 2

	visualize_img = np.ones((image_height, image_width, 3)) * 255

	viz_dir = '/mnt/c/Users/vthamizharasan/Desktop/summer_work/geom/id-only_rendered'
	viz_gt_imgs = glob.glob(os.path.join(viz_dir, '*.jpg')) 

	for name in tsne_dict.keys():
		cx, cy = tsne_dict[name]
		cx = int(cx * scatter_distance + img_cx)
		cy = int(cy * scatter_distance + img_cy)
		best_match = search_dict(name)
		if best_match == False:
			best_match = find_best_match(name.split('_0')[0] + '.jpg', viz_gt_imgs)

		print(name, best_match)
		img_gt = cv2.imread(os.path.join(viz_dir, best_match))
		img_gt = cv2.resize(img_gt,(0,0),fx=gt_image_scale,fy=gt_image_scale,interpolation=cv2.INTER_CUBIC)
		h,w,c = img_gt.shape
		odd_offset = h - (h//2)*2
		# print(cy-h//2,cy+h//2 + 1,cx-w//2,cx+w//2 + 1)
		# print(h,w,cy+h//2+1 - (cy-h//2),cx+w//2 + 1 - (cx-w//2))
		visualize_img[cy-h//2:cy+h//2 + odd_offset,cx-w//2:cx+w//2 + odd_offset,:] = img_gt

	cv2.imwrite(JSON.split('.json')[0]+'.jpg',visualize_img)

class tSNE_VIZ(object):
	def __init__(self, json_filename):
		super(tSNE_VIZ, self).__init__()
		self.fig, self.ax = plt.subplots()
		self.colors = itertools.cycle(["r", "b", "g"])
		self.checkpoint = json_filename.split('.json')[0]
		self.steps = 23
		self.count = 0
		self.bg_sz = 5
		self.active_sz = 20
		self.passive_size = 8
		self.render_img_size = 512
		self.flip = -0.5

		if not os.path.exists(self.checkpoint):
			os.makedirs(self.checkpoint)

		with open(json_filename, 'r') as fp:
			tsne_dict = json.load(fp)

		self.tsne_train_data = tsne_dict['tsne_train_data']
		self.tsne_interpolate_data = tsne_dict['tsne_interpolate_data']
		self.interpolate_pairs = tsne_dict['interpolate_pairs']

	def _savefig(self):
		self.fig.savefig(os.path.join(self.checkpoint,'%d.jpg'%(self.count)))
		self.count += 1

	def _basepoints(self):
		self.fig, self.ax = plt.subplots()
		self.size = self.fig.get_size_inches()
		self.scale = 1#self.render_img_size/640
		self.fig.set_size_inches(self.size[0]*self.scale,
								self.size[0]*self.scale)
		plt.axis('off')

		points = np.asarray(list(self.tsne_train_data.values()))
		hull = ConvexHull(points)
		for simplex in hull.simplices:
			self.ax.plot(points[simplex, 0], points[simplex, 1], 'c', color='gray')
		self.ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'o', mec='r', color='gray', lw=0.5, markersize=1)

		for subject in self.tsne_train_data.keys():
			if subject not in self.interpolate_pairs.keys():
				x,y = self.tsne_train_data[subject]
				self.ax.scatter(x,y, color="b",s=self.bg_sz,marker='x',linewidths=1)

	def plot(self):

		self._basepoints()

		x_pairs = []
		y_pairs = []
		x_line = []
		y_line = []

		i = 0
		for subject in self.interpolate_pairs.keys():
			start = subject
			end = self.interpolate_pairs[subject]
			x_s,y_s = self.tsne_train_data[start]
			x_pairs.append(x_s)
			y_pairs.append(y_s)
			self.ax.scatter(x_s,y_s, color="g",s=self.active_sz,marker='^',linewidths=2)
			self.ax.annotate("S%d"%(i), (x_s,y_s))
			x_e,y_e = self.tsne_train_data[end]
			x_pairs.append(x_e)
			y_pairs.append(y_e)
			self.ax.scatter(x_e,y_e, color="g",s=self.active_sz,marker='^',linewidths=2)
			self.ax.annotate("S%d"%(i+1), (x_e,y_e))
			self._savefig()

			_x = [x_s]
			_y = [y_s]
			for num in range(self.steps):
				x_i,y_i = self.tsne_interpolate_data['%s-%s_%d'%(start, end, num)]
				_x.append(x_i)
				_y.append(y_i)
				self.ax.plot(_x, _y, color="r", linestyle="dashed", linewidth=1)
				self.ax.scatter(x_i,y_i, color="r",s=self.active_sz/5,marker='s')
				# self.flip *= -1
				# self.ax.annotate("I%d"%(num), (x_i+self.flip,y_i))
				self._savefig()

			_x.append(x_e)
			_y.append(y_e)

			self.ax.plot(_x, _y, color="r", linestyle="dashed", linewidth=1)
			self._savefig()

			x_line += _x
			y_line += _y
			print(len(x_line), len(y_line))

			self._basepoints()
			self.ax.scatter(x_pairs,y_pairs, color="g",s=self.passive_size,marker='^',linewidths=2)
			self.ax.plot(x_line, y_line, color="gray", linestyle="dotted", linewidth=1)
			# self._savefig()

			i+=1

		# ax.set_xlabel('$c_1$')
		# ax.set_ylabel('$c_2$')

'''variables start'''
JSON = '/mnt/c/Users/vthamizharasan/Desktop/summer_work/NL_3DMM/tsne_data/\
disney_L1_KLD_geomMM_v2_neutral_captured_latent_viz_interpolate_60k.json'

FACES = '/mnt/c/Users/vthamizharasan/Desktop/summer_work/nl3dmm_results/ctx/\
disney_L1_KLD_geomMM_v2_neutral_captured_latent_viz_interpolate_60k'

TSNE_DIR = JSON.split('.json')[0]
'''end'''		

viz = tSNE_VIZ(JSON)
viz.render_img_size = 512
viz.steps = 23
# viz.plot()

# 10 pairs: 1 (activate) + 6 (intermediate) + 1 (passive) = 8

count = 0
empty_img = np.ones((viz.render_img_size, viz.render_img_size,3)) * 0
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,30)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

def multi_stack(img1,img2,img3,img4):
	global count
	ht, wd, cc = img1.shape
	wd = ht
	margin_w = np.ones((ht,2,3)) * 255
	result = np.full((ht,wd,cc), (0,0,0), dtype=np.uint8)
	xx = (wd - viz.render_img_size) // 2
	yy = (ht - viz.render_img_size) // 2
	img_comp = np.hstack((img1, margin_w))
	result[yy:yy+viz.render_img_size, xx:xx+viz.render_img_size] = img2
	img_comp = np.hstack((img_comp, result))
	# img_comp = np.hstack((img_comp, margin_w))
	result[yy:yy+viz.render_img_size, xx:xx+viz.render_img_size] = img3
	img_comp = np.hstack((img_comp, result))
	# img_comp = np.hstack((img_comp, margin_w))
	result[yy:yy+viz.render_img_size, xx:xx+viz.render_img_size] = img4
	img_comp = np.hstack((img_comp, result))
	cv2.imwrite(os.path.join(FACES,'%d_comb.jpg'%(count)), img_comp)
	count += 1

i = 0
subject_id = 0
for subject in viz.interpolate_pairs.keys():
	start = subject
	end = viz.interpolate_pairs[subject]
	active = cv2.imread(os.path.join(TSNE_DIR,'%d.jpg')%(i))
	i+=1
	img_1_gt = cv2.imread(os.path.join(FACES,'%s_gt_face.jpg'%(start)))	
	img_2_gt = cv2.imread(os.path.join(FACES,'%s_gt_face.jpg'%(end)))

	cv2.putText(img_1_gt,'S%d'%(subject_id), 
	    bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
	cv2.putText(img_2_gt,'S%d'%(subject_id+1), 
	    bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

	# img_comp = multi_stack(active, img_1_gt, img_1_gt, img_2_gt)
	

	for num in range(viz.steps):
		intermediate = cv2.imread(os.path.join(TSNE_DIR,'%d.jpg')%(i))
		i+=1
		img_interpolate = cv2.imread(os.path.join(FACES,'%s-%s_%d_predicted_face.jpg'%(start,end,num)))	
		cv2.putText(img_interpolate,'I%d'%(num), 
	    	bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
		img_comp = multi_stack(intermediate, img_1_gt, img_interpolate, img_2_gt)

	passive = cv2.imread(os.path.join(TSNE_DIR,'%d.jpg')%(i))
	i+=1
	# img_comp = multi_stack(passive, img_1_gt, img_2_gt, img_2_gt)

	subject_id+=1

