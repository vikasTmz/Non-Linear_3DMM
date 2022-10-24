'''
'''
import bpy
from bpy import context, data, ops
import sys
import os
import argparse 
import math
import random
import time
import glob
import json

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

class Blender():
	def __init__(self):
		self.model = None
		self.ob = None
		self.width = 512
		self.height = 512
	def use_gpu(self):
		'''
			Set device to GPU
		'''
		bpy.context.scene.cycles.device = 'GPU'
		prefs = bpy.context.preferences
		cprefs = prefs.addons['cycles'].preferences
		cprefs.compute_device_type = 'CUDA'
		cprefs.get_devices()
		for device in cprefs.devices:
			device.use = True
		bpy.ops.wm.save_userpref()
	def import_obj(self, path):
		'''
			Imports .obj model
		'''
		old_obj = set(context.scene.objects)
		bpy.ops.import_scene.obj(filepath=path, filter_glob="*.obj")
		self.model = (set(context.scene.objects) - old_obj).pop()
		self.select_object()
	def import_ply(self, path):
		'''
			Imports .obj model
		'''
		old_obj = set(context.scene.objects)
		bpy.ops.import_mesh.ply(filepath=path, filter_glob="*.ply")
		self.model = (set(context.scene.objects) - old_obj).pop()
		self.select_object()
	def remove_object(self):
		self.select_object()
		bpy.ops.object.delete()
	def select_object(self):
		objs = bpy.data.objects
		for i in objs:
			if i.name not in ['Camera','Point','Point.001','Point.002']:
				self.model_name = i.name
				self.ob = bpy.data.objects[i.name]
				self.ob.select_set(True)
	def transform_model(self):
		'''
			Scales the loaded object,
			applies smoothing to it,
			applies rotation to the object
		'''
		self.select_object()
		# Apply smooth shading
		mesh = self.ob.data
		for f in mesh.polygons:
			f.use_smooth = True
		# Scale down face
		self.ob.scale = (0.005,0.005,0.005)
		self.ob.rotation_euler = (90*math.pi/180,0,0)
		bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')
	def set_material(self, material):
		'''
			Set the pre-defined materials: Diffuse
		'''
		self.select_object()
		context.view_layer.objects.active = self.ob
		if material in bpy.data.materials:
			mat = bpy.data.materials[material]
			if self.ob.data.materials:
				self.ob.data.materials[0] = mat
			else:
				self.ob.data.materials.append(mat)
		print("Assigned materials")
	def set_albedo(self, albedo_path, albedo_name, material):
		'''
			Load albedo texture map
		'''
		for image in bpy.data.images:
			if image.name.split('.png')[0] == albedo_name:
				bpy.data.images.remove(image)
		bpy.data.images.load(albedo_path, check_existing=False)
		mat = bpy.data.materials[material]
		bsdf = mat.node_tree.nodes['Image Texture']
		bsdf.image = bpy.data.images[albedo_name]
	def render_components(self, modes, output_pth, name):
		for mode in modes:
			self.set_material(mode)
			self.save_render(mode, output_pth, name)
	def save_render(self, mode, output_pth, name):
		# self.use_gpu()
		# context.scene.cycles.device = 'GPU'
		# context.scene.cycles.samples = samples
		context.scene.render.resolution_x = self.width
		context.scene.render.resolution_y = self.height
		context.scene.render.resolution_percentage = 100
		bpy.context.scene.node_tree.nodes["File Output"].base_path = output_pth
		bpy.ops.render.render() # write_still=True
		os.system('move ' + os.path.join(output_pth,'image0003.jpg') + ' ' + name)

start = time.time()

EXTENSION = '.obj'
modes = ["DiffuseTex"] #Diffuse,DiffuseTex
TM_PREFFIX = ''
TM_SUFFIX = '.png'

for checkpoints in ['disney_L1_KLD_V2_albedoMM_neutral_only_version2.0_itersampleVAE_60k']:
	input_path = 'C:\\Users\\vthamizharasan\\Desktop\\summer_work\\nl3dmm_results\\ctx_albedo\\' + checkpoints
	input_objs = glob.glob(os.path.join(input_path,'*' + EXTENSION))
	input_objs.sort()
	resolution = '512x512'
	blender_instance = Blender()
	importer = {'.ply':blender_instance.import_ply,'.obj':blender_instance.import_obj}
	blender_instance.width = int(resolution.split('x')[0])
	blender_instance.height = int(resolution.split('x')[1])
	for obj_path in input_objs:
		try:
			print(obj_path)
			# Import OBJ
			importer[EXTENSION](obj_path)
			# Transform the imported face
			blender_instance.transform_model()
			# Set Albedo Map
			if modes[0] == "DiffuseTex":
				albedo_name = TM_PREFFIX + obj_path.split('\\')[-1].split(EXTENSION)[0] + TM_SUFFIX
				albedo_path = os.path.join(input_path,albedo_name)
				print(albedo_path)
				blender_instance.set_albedo(albedo_path, albedo_name, modes[0])
			# Render components
			blender_instance.render_components(modes, input_path, obj_path.split(EXTENSION)[0] + '_render_tex.jpg')
			# Remove object
			blender_instance.remove_object()
		except:
			print("An error occured")

print(time.time() - start)

