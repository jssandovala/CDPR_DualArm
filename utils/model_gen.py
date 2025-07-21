from lxml import etree
import numpy as np

def gen_mp_file(model_dict):
	filename  = model_dict["mp_file"]
	og_pos    = model_dict["og_pos"]
	mp_size_l = model_dict["mp_size_l"]
	ei_pos   = model_dict["ei_pos"]

	mp_size = ' '.join([str(0.5*el) for el in mp_size_l])

	data = etree.Element('mujoco')

	worldbody = etree.SubElement(data, 'worldbody')

	og_pos_s = ' '.join([str(el) for el in og_pos])
	
	body = etree.SubElement(worldbody, 'body')
	body.set('name', 'MP')
	body.set('pos', og_pos_s)

	site = etree.SubElement(body, 'site')
	site.set('name','mp_center')
	site.set('type', 'sphere')
	site.set('pos', "0 0 0")
	site.set('size', '0.01')
	site.set('rgba', '1 1 0 1')


	joint = etree.SubElement(body, 'joint')
	joint.set('type','slide')
	joint.set('name','mp_joint_1')
	joint.set('axis','1 0 0')

	joint = etree.SubElement(body, 'joint')
	joint.set('type','slide')
	joint.set('name','mp_joint_2')
	joint.set('axis','0 1 0')

	joint = etree.SubElement(body, 'joint')
	joint.set('type','slide')
	joint.set('name','mp_joint_3')
	joint.set('axis','0 0 1')
	
	joint = etree.SubElement(body, 'joint')
	joint.set('type','hinge')
	joint.set('name','mp_joint_4')
	joint.set('axis','1 0 0')

	joint = etree.SubElement(body, 'joint')
	joint.set('type','hinge')
	joint.set('name','mp_joint_5')
	joint.set('axis','0 1 0')

	joint = etree.SubElement(body, 'joint')
	joint.set('type','hinge')
	joint.set('name','mp_joint_6')
	joint.set('axis','0 0 1')


	geom = etree.SubElement(body, 'geom')
	geom.set('name','mp_frame')
	geom.set('type','box')
	geom.set('size',mp_size)
	geom.set('pos','0 0 0')
	geom.set('rgba','1 1 1 1')


	
	ei_list = []
	for i in range(8):
		ei_pos_i = ' '.join([str(el) for el in ei_pos[i]])

		ei_list.append(etree.SubElement(body, 'site'))
		ei_list[-1].set('name',"mp_ei_"+str(i+1))
		ei_list[-1].set('type','sphere')
		ei_list[-1].set('size',"0.01")
		ei_list[-1].set('pos',ei_pos_i)
		ei_list[-1].set('rgba','1 1 0 1')

	interial = etree.SubElement(body, 'inertial')
	interial.set('mass','1e-3')
	interial.set('diaginertia',"1e-4 1e-4 1e-4")
	interial.set('pos','0 0 0')


	file = open(filename,'wb')
	file.write(etree.tostring(data,pretty_print=True))
	file.close()

def gen_cb_file(model_dict):
	ai_pos   = model_dict["ai_pos"]
	filename = model_dict["cable_file"]

	data = etree.Element('mujoco')

	worldbody = etree.SubElement(data, 'worldbody')


	for i in range(8):

		ai_pos_i = ' '.join([str(el) for el in ai_pos[i]])

		body = etree.SubElement(worldbody,'body')
		body.set('name','root_ball_'+str(i+1))
		body.set('pos', ai_pos_i)

		site = etree.SubElement(body, 'site')
		site.set('name', 'ai_site_'+str(i+1))
		site.set('type', 'sphere')
		site.set('pos', "0 0 0")
		site.set('size', '0.05')
		site.set('rgba', '1 0 0 1')


	file = open(filename,'wb')
	file.write(etree.tostring(data,pretty_print=True))
	file.close()

def gen_main_file(model_dict):
	include_files = model_dict["include_files"]
	filename      = model_dict["main_file"]

	data = etree.Element('mujoco')

	for el in include_files:
		include1 = etree.SubElement(data, 'include')
		include1.set('file',el)
	

	compiler = etree.SubElement(data, 'compiler')
	compiler.set('autolimits','true')


	defaults = etree.SubElement(data, 'default')
	
	site_def = etree.SubElement(defaults, 'site')
	site_def.set('type','sphere')
	site_def.set('rgba',"1 0 0 1")
	site_def.set('size','0.005')

	tendon_def = etree.SubElement(defaults, 'tendon')
	tendon_def.set('rgba',"0 1 0 1")


	visual = etree.SubElement(data, 'visual')
	
	headlight = etree.SubElement(visual, 'headlight')
	headlight.set('diffuse','.7 .7 .7')
	
	actuator = etree.SubElement(data, 'actuator')
	sensor = etree.SubElement(data, 'sensor')
	tendon = etree.SubElement(data, 'tendon')


	mp_vel = etree.SubElement(sensor, 'velocimeter')
	mp_vel.set('name', 'mp_vel')
	mp_vel.set('site', 'mp_center')

	mp_gyro = etree.SubElement(sensor, 'gyro')
	mp_gyro.set('name', 'mp_gyro')
	mp_gyro.set('site', 'mp_center')

	mp_ctrl_range = '-20 20'

	for i in range(6):
		intvelocity = etree.SubElement(actuator, 'velocity')
		intvelocity.set('name', 'mp_joint_'+str(i+1))
		intvelocity.set('joint', 'mp_joint_'+str(i+1))
		intvelocity.set('ctrlrange', mp_ctrl_range)
		intvelocity.set('kv', '1e7')

		
	
	for i in range(8):

		spatial = etree.SubElement(tendon,'spatial')
		spatial.set('name','rope_'+str(i+1))
		spatial.set('width','0.001')
		spatial.set('stiffness','0')

		site = etree.SubElement(spatial,'site')
		site.set('site','ai_site_'+str(i+1))

		site = etree.SubElement(spatial,'site')
		site.set('site','mp_ei_'+str(i+1))
	

	file = open(filename,'wb')
	file.write(etree.tostring(data,pretty_print=True))
	file.close()

def gen_files(data_dict):
	gen_mp_file(data_dict)
	gen_cb_file(data_dict)
	gen_main_file(data_dict)

