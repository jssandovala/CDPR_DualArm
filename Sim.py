import mujoco
import mujoco.viewer


import numpy as np
import math
from time import perf_counter,sleep

from Controller import *
from utils.model_gen import *
from utils.cdpr_utils import *
from scipy.spatial.transform import Rotation as R


class Sim:
  def __init__(self,model_dict):
    
    self.viewer = None

    self.model_dict = model_dict
    #gen_files(self.model_dict)

    self.sim_model = mujoco.MjModel.from_xml_path(self.model_dict['main_file'])
    self.sim_data = mujoco.MjData(self.sim_model)

    self.controller = Controller(self.model_dict)
    self.controller.sim = self
    self.controller.model = self.sim_model      # <-- add these two
    self.controller.data  = self.sim_data 
    self.controller.dt    = self.sim_model.opt.timestep 

    self.reset_vars()

    #self.target_marker_id = self.sim_model.body("target_marker").mocapid



  def isclose(self, x, y, rtol=1.e-5, atol=1.e-8):
      return abs(x-y) <= atol + rtol * abs(y)

  def get_orientation(self,site):
    R = site.xmat.reshape((3,3))

    theta_z = 0.0
    if self.isclose(R[2,0],-1.0):
        theta_y = math.pi/2.0
        theta_x = math.atan2(R[0,1],R[0,2])
    
    elif self.isclose(R[2,0],1.0):
        theta_y = -math.pi/2.0
        theta_x = math.atan2(-R[0,1],-R[0,2])
    
    else:
        theta_y = -math.asin(R[2,0])
        cos_theta = math.cos(theta_y)
        theta_x = math.atan2(R[2,1]/cos_theta, R[2,2]/cos_theta)
        theta_z = math.atan2(R[1,0]/cos_theta, R[0,0]/cos_theta)
    
    return [theta_x, theta_y, theta_z]

  def get_pose(self,site):
    X = np.zeros((6,))
    X[:3] = site.xpos
    X[3:] = self.get_orientation(site)

    return X
  
  def get_x_xp(self):

    x = self.get_pose(self.sim_data.site('mp_center'))
    xp = np.zeros((6,))

    for i in range(3):
       xp[i] = self.sim_data.sensor('mp_vel').data[i]
       xp[i+3] = self.sim_data.sensor('mp_gyro').data[i]
    
    
    return x,xp
  def get_kuka_ee_pose(self):
    pos = np.copy(self.sim_data.site('ee_site').xpos)
    rot = self.get_orientation(self.sim_data.site('ee_site'))
    return np.concatenate((pos, rot))  # 6D pose: [x y z rx ry rz]

  def get_cable_lengths(self):
      lengths = np.zeros((8,))
      for i in range(8):
          lengths[i] = self.sim_data.ten_length[i]
      return lengths

    
  def get_full_q_vector(self):
      # Get world pose of MP from the site
      mp_pose = self.get_pose(self.sim_data.site('mp_center'))  # [x, y, z, roll, pitch, yaw]

      # Get current KUKA joint angles from qpos[6:]
      kuka_q = np.copy(self.sim_data.qpos[6:])  # 7 joint values

      # Combine into single 13D vector
      q = np.concatenate((mp_pose, kuka_q))

      return q

  def apply_wrench(self,wrench):
    self.applied_wrench = wrench
    # self.sim_data.xfrc_applied[self.sim_model.body('MP').id] = wrench

  def reset_vars(self):
    self.sim_open = True
    self.sim_fast = False
    self.sim_run = False

    self.joint_reset = False
    self.damping_reset = False
    self.did_print_debug = False

    self.applied_wrench = np.zeros((6,))

    self.sim_time = 0
    self.sim_time_last = 0
    self.sim_view_time_last = 0
    self.sim_view_step = 0.015
    self.control_step = 1./500.
    self.contrl_time_last = 0

    self.sim_model.opt.integrator = 2


  def init_sim(self,pos=None):
    self.reset_vars()

    mujoco.mj_resetData(self.sim_model, self.sim_data)
    self.sim_data.qvel[:] = np.random.randn(self.sim_data.qvel.size) * 1e-5
    self.sim_data.ctrl[:] = np.zeros(self.sim_model.nu)


    # --- Set initial KUKA joint angles here ---
    for i in range(self.sim_model.njnt):
      name = name = self.sim_model.joint(i).name

      qpos_adr = self.sim_model.jnt_qposadr[i]
      print(f"Joint {i}: {name}, qpos_adr = {qpos_adr}")




    kuka_joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint",
        "elbow_joint", "wrist_1_joint", "wrist_2_joint","wrist_3_joint"
    ]

    #initial_joint_angles = [0.0, -1, 1, 1, -1, 1.5]  # in radians
    #initial_joint_angles = [0.0, 0, 0, 0.0, 0, 0.0]


    #initial pose of the arm
    initial_joint_angles = [0.0, 0, -1, 0, 1, 0]
    

    for name, angle in zip(kuka_joint_names, initial_joint_angles):
        qpos_adr = self.sim_model.joint(name).qposadr
        dof_adr = self.sim_model.joint(name).dofadr  # This maps to velocity
        self.sim_data.qpos[qpos_adr] = angle         # Set position
        self.sim_data.qvel[dof_adr] = 0.0            # Set velocity to 0
        self.sim_data.ctrl[8 + kuka_joint_names.index(name)] = 0.0  # Pre-fill ctrl with neutral


    if pos is not None:
      self.sim_model.body('MP').pos = pos


    X = self.get_pose(self.sim_data.site('mp_center'))
    X[:3] = self.sim_model.body('MP').pos

    self.controller = Controller(self.model_dict, is_ptp=False)
    self.controller.sim = self
    self.controller.model = self.sim_model      # add
    self.controller.data  = self.sim_data   
    print("[DEBUG] Controller.sim set:", hasattr(self.controller, "sim"))

    x,xp = self.get_x_xp()
    self.controller.update_xmes(x,xp)
    self.controller.update_targ(X,np.zeros((6)))
    




    self.sim_run = True


  #this function serves to send small velocity commands to keep the arm in place when the trajectory is not running, otherwise gravity would bring it down
  def hold_kuka_pose(self, target_angles, kp=1.0, kd=0.0, max_vel=2.0):
      joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint",
        "elbow_joint", "wrist_1_joint", "wrist_2_joint","wrist_3_joint"
    ]

      if not hasattr(self, "_last_qpos"):
          self._last_qpos = np.zeros(len(joint_names))

      arm_vel_cmd = np.zeros(6)

      for i, name in enumerate(joint_names):
          j_id = self.sim_model.joint(name).id
          qpos_adr = self.sim_model.joint(name).qposadr
          dof_adr = self.sim_model.joint(name).dofadr

          q = float(self.sim_data.qpos[qpos_adr])     # current position
          dq = float(self.sim_data.qvel[dof_adr])     # current velocity
          error = target_angles[i] - q
          d_error = -dq
          


          # PD velocity command
          vel_cmd = kp * error + kd * d_error
          vel_cmd = np.clip(vel_cmd, -max_vel, max_vel)
          arm_vel_cmd[i] = vel_cmd  # 

          # Update internal memory
          delta_q = q - self._last_qpos[i]
          self._last_qpos[i] = q

          #print(error)

      return arm_vel_cmd  # ⬅️ return it cleanly





    
  def launch_sim(self):
    
    self.viewer = mujoco.viewer.launch_passive(self.sim_model, self.sim_data)
    self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
    self.viewer.sync()

    # sleep(5)
    self.init_sim()



  
    while self.sim_open:

      if self.sim_run:

        self.sim_time = perf_counter()

        if self.sim_time-self.contrl_time_last > self.control_step:
            self.contrl_time_last = self.sim_time

            x,xp = self.get_x_xp()
            self.controller.update_xmes(x,xp)
            self.controller.update_cmd()
            

            


            self.sim_data.ctrl = self.controller.vel_cmd




            l_mes = self.get_cable_lengths()
            self.controller.update_lmes(l_mes)


            
            



        
        if self.sim_time-self.sim_time_last > self.sim_model.opt.timestep or self.sim_fast:
          
          self.sim_time_last = self.sim_time
          mujoco.mj_step(self.sim_model, self.sim_data)
          
          
        
        
        if self.sim_time-self.sim_view_time_last > self.sim_view_step:
          self.sim_view_time_last = self.sim_time
          self.viewer.sync()
          # 🧠 Add this below:
          if not self.did_print_debug and self.sim_time > 0.5:
              self.did_print_debug = True

              ee_pose = self.get_kuka_ee_pose()
              q = self.get_full_q_vector()

              # Get predicted end-effector transform
              #T_pred = self.controller.mujoco_forward_kinematics(q)
              T_pred = self.controller.compute_total_forward_kinematics(q)
              pos_pred = T_pred[:3, 3]
              # Convert rotation matrix to euler for comparison
              from scipy.spatial.transform import Rotation as R
              r_pred = R.from_matrix(T_pred[:3, :3]).as_euler('xyz')


              R_fk = T_pred[:3, :3]       # From your FK
              R_sim = R.from_euler('xyz', [-1.571, 0.0, 3.142]).as_matrix()
              print("rotation thing: ",np.allclose(R_fk, R_sim, atol=1e-3))



              pos_error = np.linalg.norm(ee_pose[:3] - pos_pred)
              ori_error = np.linalg.norm(np.unwrap(ee_pose[3:] - r_pred))  # unwrap to avoid π-wrap jumps
              J_test = self.controller.geometric_jacobian(q)
              print("Z from FK:", T_pred[2, 3])
              print("Z from MuJoCo:", self.sim_data.site('ee_site').xpos[2])

              print(f"[EE POSE] x={ee_pose[0]:.3f}, y={ee_pose[1]:.3f}, z={ee_pose[2]:.3f}, "
                    f"rx={ee_pose[3]:.3f}, ry={ee_pose[4]:.3f}, rz={ee_pose[5]:.3f}")
              print(f"[FK PRED]  x={pos_pred[0]:.3f}, y={pos_pred[1]:.3f}, z={pos_pred[2]:.3f}, "
                    f"rx={r_pred[0]:.3f}, ry={r_pred[1]:.3f}, rz={r_pred[2]:.3f}")
              print(f"[qpos] {q}")
              print(f"[DEBUG][PTP] pos_err: {pos_error:.4f}, ori_err: {ori_error:.4f}")
              print("")


      else:
         sleep(0.1)
    

  def close(self):
     self.sim_open = False
     self.viewer.close()
     