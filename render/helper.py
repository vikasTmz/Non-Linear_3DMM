cheekPuffL.obj
cheekSquintL.obj
cheekSquintR.obj
chinRaiser.obj
chinRaiserLowerL.obj
chinRaiserLowerR.obj
dimplerR.obj
dimplerL.obj
eyeCloseL.obj
eyeCloseR.obj
eyeCompressL.obj
eyeCompressR.obj
funnelLowerL.obj
funnelLowerR.obj
funnelUpperL.obj
funnelUpperR.obj
jawOpen.obj
jawOpen_lipCornerPullerL.obj
jawOpen_lipCornerPullerR.obj
jawOpen_lipsTogether_lipCornerPullerL.obj
jawOpen_lipsTogether_lipCornerPullerR.obj
jawOpen_lipsTogether_upperLipRaiserL.obj
jawOpen_lipsTogether_upperLipRaiserR.obj
jawright.obj
lipCornerPullerL.obj
lipCornerPullerR.obj
lipMidLowerDownTweak.obj
lipMidUpperUpTweak.obj
lipOpenMoreLowerL.obj
lipOpenMoreLowerR.obj
lipStretcherL.obj
lipStretcherR.obj
mouthSwingL.obj
mouthSwingR.obj
noseWrinklerR.obj
upperLipRaiserR.obj
upperLipRaiserL.obj

import cv2
import numpy as np

subj = ['C_MEKERTICHYAN_26','A_NICOL_32','B_NABIPOUR_29','C_JOYCE_11','B_ANTONIAZZI_28','D_WILLIAMS_36','J_TANNER_2','D_SAVAGE_21','J_FAHSELT_34','M_SILVERSTEN_9']

repo = ['disney_L1_train_interpolate_40k']

for s in subj:
	img_gt = cv2.imread('ctx/%s/%s_gt_face.jpg'%(repo[0],s))
	margin_b = np.ones((512, 20,3))*255
	margin_w = np.ones((512, 60,3)) * 255
	img_comp = np.hstack((img_gt, margin_w))
	for i,d in enumerate(repo):
		print(d)
		img_d = cv2.imread('ctx/%s/%s_predicted_face.jpg'%(d,s))	
		img_comp = np.hstack((img_comp, img_d))
		img_comp = np.hstack((img_comp, margin_b))
	cv2.imwrite('ctx/%s_comb.jpg'%(s), img_comp)


import cv2
import numpy as np
repo = ['disney_L1_albedoMM_full_interpolate_50k']
root = 'ctx_albedo'

pairs = ['A_ZHUKOV_0-A_REZAEE_0','A_REZAEE_0-J_JIMENEZ_0','J_JIMENEZ_0-H_BANSAL_0','H_BANSAL_0-M_HO_0',\
				'M_HO_0-H_SADLER_0','H_SADLER_0-A_ELLIOT_0','A_ELLIOT_0-A_ZHUKOV_0']

for j in range(0,len(pairs)):
	for d in repo:
		img_gt_1 = cv2.imread('%s/%s/%s_%s_gt_face.jpg'%(root,d,pairs[j],pairs[j].split('-')[0]))
		img_gt_2 = cv2.imread('%s/%s/%s_%s_gt_face.jpg'%(root,d,pairs[j],pairs[j].split('-')[1]))
		for i in range(0,30):
			img_d = cv2.imread('%s/%s/%s_%d_predicted_face.jpg'%(root,d,pairs[j],i))	
			margin_w = np.ones((512, 60,3)) * 255
			img_comp = np.hstack((img_gt_1, margin_w))
			img_comp = np.hstack((img_comp, img_d))
			img_comp = np.hstack((img_comp, margin_w))
			img_comp = np.hstack((img_comp, img_gt_2))
			cv2.imwrite('%s/%s/%d_comb.jpg'%(root,d,i+30*(j+1)), img_comp)

# new
count = 0
for i in [0,6,10,11,14,16,21,28,31]:
	for j in range(0,5):
		margin_w = np.ones((512, 60,3)) * 255
		img_neutral = cv2.imread('R_ALEXANDER_rot45_0_tuning_0.jpg')	
		font                   = cv2.FONT_HERSHEY_SIMPLEX
		bottomLeftCornerOfText = (10,30)
		fontScale              = 1
		fontColor              = (255,255,255)
		lineType               = 1
		cv2.putText(img_neutral,'parameter %d/32'%(i+1), 
		    bottomLeftCornerOfText, 
		    font, 
		    fontScale,
		    fontColor,
		    lineType)
		img_1 = cv2.imread('R_ALEXANDER_rot45_%d_tuning_%d.jpg'%(i,j))	
		img_2 = cv2.imread('R_ALEXANDER_rot0_%d_tuning_%d.jpg'%(i,j))	
		img_comp = np.hstack((img_neutral, margin_w))
		img_comp = np.hstack((img_comp, img_1))
		img_comp = np.hstack((img_comp, margin_w))
		img_comp = np.hstack((img_comp, img_2))
		cv2.imwrite('%d_comb.jpg'%(count), img_comp)
		count += 1

convert                                                  \
  -delay 10                                              \
   $(for i in $(seq 0 1 82); do echo ${i}.png; done) \
  -loop 0                                               \
  plot.gif

$(for i in $(seq 0 1 31);do for j in $(seq 0 1 4);\
	do mv G_BRAMBLE_0_param${i}_${j}_predicted_face.jpg G_BRAMBLE_rot45_${i}_tuning_${j}.jpg;
	done; done)

convert                                                  \
  -delay 30                                             \
   $(for i in $(seq 1 1 3); do echo identity_space_param_tuning${i}.gif; done) \
  -loop 0                                                \
   identity_space_param_tuning.gif


$(for j in $(seq 0 1 10);do mv ${j}.png head_0${j}.png;done)


cd disney_L1_KLD_geomMM_v2_neutral_captured_30k_latent_viz_interpolate;\
convert -delay 20 $(for i in $(seq 0 1 82); do echo ${i}.jpg; done) -loop 0 plot_1.gif;\
cd ../


convert -delay 11 $(for i in $(seq 0 1 70); do echo ${i}_comb.jpg; done) -loop 1 plot1.gif;
convert -delay 11 $(for i in $(seq 71 1 175); do echo ${i}_comb.jpg; done) -loop 1 plot2.gif;
convert -delay 11 $(for i in $(seq 1 1 2); do echo plot${i}.gif; done) -loop 0 plot.gif;