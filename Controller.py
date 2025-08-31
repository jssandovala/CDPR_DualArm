from time import sleep
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

from qpsolvers import solve_qp
import numpy as np
import mujoco

from utils.traj_gen import *
from utils.cdpr_utils import CableJacobian
from scipy.spatial.transform import Rotation as R
from scipy.sparse import csc_matrix

class Controller:

    def __init__(self,model_dict, is_ptp=False):
        self.x_targ = np.zeros((6,))
        self.xp_targ = np.zeros((6,))
        self.vel_cmd = np.zeros((15,))

        self.debug_mode = False  # enables/disables printing in update_cmd

        self.synced = False
        self.is_ptp = is_ptp

        #kpx = 10*np.eye(3)          #for ptp 10
        #kpt = 10*np.eye(3)
        #kpt = np.diag([100, 100, 200]) #for ptp [200, 200, 400]
        if self.is_ptp:
            kpx = 5 * np.eye(3)
            kpt = np.diag([100, 100, 200])
        else:
            kpx = 10 * np.eye(3)
            kpt = np.diag([100, 100, 200])


        self.Kpx = block_diag(kpx,kpt)

        self.dt = 1./500
        self.model_dict = model_dict

        self.l_mes = np.zeros((8,))
        self.l_des = np.zeros((8,))
        self.l_cmd = np.zeros((8,))
        #self.Kp_l = 50.0  # Tune this
        self.Kp_l = np.array([
            [50, 0,  0,  0,  0,  0,  0,  0],
            [0,  50, 0,  0,  0,  0,  0,  0],
            [0,  0,  50, 0,  0,  0,  0,  0],
            [0,  0,  0,  50, 0,  0,  0,  0],
            [0,  0,  0,  0,  50, 0,  0,  0],
            [0,  0,  0,  0,  0,  50, 0,  0],
            [0,  0,  0,  0,  0,  0,  50, 0],
            [0,  0,  0,  0,  0,  0,  0, 50]
        ])

        self.Kd_l = 0.0  # Optional for now
        # Modified DH parameters: (alpha, a, d, theta_offset)
        self.dh_params = [
            (0,          0,       0.181,       0),            # Joint 1
            (-np.pi/2,   0,       0,            0),            # Joint 2
            (0,          0.612,   0,            0),            # Joint 3
            (0,          0.5723,  0.1639,       0),            # Joint 4
            (-np.pi/2,   0,       0.1157,       0),            # Joint 5
            (np.pi/2,    0,       0,            np.pi/2),      # Joint 6 (offset π/2)
            (0,          0,       0.15,         0),            # Fixed EE offset
        ]


        self.running_traj = False  
        self._debug_counter = 0
        self._debug_max = 20
        self.n_joints = 13
        self.prev_vel_cmd = np.zeros((13,))  

        self.error_history = []
        self.time_history = []
        self._ptp_start_time = None

        self._ik_converged = False

        


    def update_lmes(self, l_mes):
        self.l_mes = l_mes


    def update_xmes(self,x_mes,xp_mes):
        self.x_mes = x_mes
        self.xp_mes = xp_mes

    def update_targ(self,x_targ,xp_targ):
        self.x_targ = x_targ
        self.xp_targ = xp_targ
        #self._ik_converged = False

    """""
    def update_cmd(self):
        from utils.cdpr_utils import IGM

        # Step 1: Compute desired cable lengths analytically from target pose
        self.l_des = IGM(self.x_targ, self.model_dict)

        # Step 2: PD control (if you have velocity feedback, otherwise just P)
        error_l = self.l_des - self.l_mes
        self.l_cmd = self.Kp_l @ error_l  # + self.Kd_l * (l_des_dot - l_mes_dot), if available

        # Step 3: Send to actuators
        self.vel_cmd = self.l_cmd.copy()

        # ---- DEBUG ----
        if self.running_traj and self._debug_counter < self._debug_max:
            print("[DEBUG] x_mes:", self.x_mes.round(3))
            print("[DEBUG] x_targ:", self.x_targ.round(3))
            print("[DEBUG] l_mes:", self.l_mes.round(4))
            print("[DEBUG] l_des:", self.l_des.round(4))
            print("[DEBUG] vel_cmd:", self.vel_cmd.round(4))
            print()
            self._debug_counter += 1
    """""
    from scipy.spatial.transform import Rotation as R
    # ───────────────────────────────────────────────────────────────
    # 1. Mobile Base Transform
    # ───────────────────────────────────────────────────────────────
    def compute_mobile_transform(self, x, y, z, roll, pitch, yaw):
        """
        Computes the 4x4 transform of the mobile platform in world frame.
        Rotation is XYZ = roll, pitch, yaw.
        """
        T = np.eye(4)
        T[:3, 3] = [x, y, z]
        T[:3, :3] = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        return T

    # ───────────────────────────────────────────────────────────────
    # 2. Orbit Joint Transform (MP → arm_orbit_link)
    # ───────────────────────────────────────────────────────────────
    def compute_T_orbit(self, theta_orbit):
        """
        Computes the transform from the platform (MP) to the orbit link.
        """
        c, s = np.cos(theta_orbit), np.sin(theta_orbit)
        T = np.eye(4)
        T[:3, 3] = [0, 0, -0.19]  # from pos="0 0 -0.19"
        T[:3, :3] = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        return T

    # ───────────────────────────────────────────────────────────────
    # 3. Static Transform (arm_orbit_link → arm_mount)
    # ───────────────────────────────────────────────────────────────
    def get_T_offset(self):
        """
        Static transform from orbit link to arm mount.
        Includes 0.7m offset and 180° Y-axis rotation.
        """
        T = np.eye(4)
        T[:3, 3] = [0.7, 0, 0]
        T[:3, :3] = np.array([
            [-1, 0,  0],
            [ 0, 1,  0],
            [ 0, 0, -1]
        ])
        return T

    # ───────────────────────────────────────────────────────────────
    # 4. Modified DH Transform
    # ───────────────────────────────────────────────────────────────
    def compute_dh_transform(self, alpha, a, d, theta):
        """
        Returns the 4x4 modified DH transform using Craig convention.
        """
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)

        return np.array([
            [ct, -st, 0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0, 0, 0, 1]
        ])

    # ───────────────────────────────────────────────────────────────
    # 5. Compute All Relative Arm Transforms (joint i-1 → i)
    # ───────────────────────────────────────────────────────────────
    def compute_arm_fk_all(self, q_arm):
        """
        Returns a list of relative transforms from joint i-1 to joint i
        using self.dh_params.
        """
        T_list = []
        for i, (alpha, a, d, theta_offset) in enumerate(self.dh_params):
            if i < len(q_arm):
                theta = q_arm[i] + theta_offset
            else:
                theta = theta_offset
            T_i = self.compute_dh_transform(alpha, a, d, theta)
            T_list.append(T_i)
        return T_list

    # ───────────────────────────────────────────────────────────────
    # 6. Full Forward Kinematics
    # ───────────────────────────────────────────────────────────────
    def compute_total_forward_kinematics(self, q):
        """
        Computes the full forward kinematics of:
        [mobile base] + [orbit joint] + [arm mount] + [UR10 arm].
        
        q: [x, y, z, roll, pitch, yaw, theta_orbit, q1...q6]
        """
        # Parse joint vector
        x, y, z = q[0:3]
        roll, pitch, yaw = q[3:6]
        theta_orbit = q[6]
        q_arm = q[7:]

        # Mobile platform transform
        T_base = self.compute_mobile_transform(x, y, z, roll, pitch, yaw)

        # Orbit joint + mount transform
        T_down = np.eye(4)
        T_down[:3, 3] = [0, 0, -0.19]  # from pos="0 0 -0.19"

        T_orbit = np.eye(4)
        T_orbit[:3, :3] = R.from_euler('z', theta_orbit).as_matrix()  # orbit joint rotation

        T_mount = np.eye(4)
        T_mount[:3, 3] = [0.7, 0, 0]  # from pos="0.7 0 0"

        T_mount_correction = np.array([
                                    [1,  0,  0, 0],
                                    [0, -1,  0, 0],
                                    [0,  0, -1, 0],
                                    [0,  0,  0, 1]
                                ])


        T_yflip = T_mount_correction  # manually enforced reflection


        # Final transform to mount
        T_orbital = T_down @ T_orbit @ T_mount @ T_yflip

        # Arm FK
        T_arm = np.eye(4)
        T_list = self.compute_arm_fk_all(q_arm)
        for T_rel in T_list:
            T_arm = T_arm @ T_rel

        # Optionally add TCP offset if your DH chain ends at wrist 3
        T_tcp = self.compute_dh_transform(0, 0, 0.01, np.pi/2)
        T_arm = T_arm @ T_tcp

        # Total FK
        T_total = T_base @ T_orbital @ T_arm
        return T_total


    # ───────────────────────────────────────────────────────────────
    # 7. Full Analytical Jacobian
    # ───────────────────────────────────────────────────────────────
    def geometric_jacobian(self, q):
        """
        Computes the 6x13 analytical Jacobian for:
        [mobile base (6)] + [orbit joint (1)] + [UR10 arm (6)]
        Returns the spatial Jacobian (linear + angular twist).
        """
        from scipy.spatial.transform import Rotation as R

        # === Parse joint vector ===
        x, y, z = q[0:3]
        roll, pitch, yaw = q[3:6]
        theta_orbit = q[6]
        q_arm = q[7:]
        n_total = len(q)

        # === Build transforms (same as FK) ===
        # Mobile base
        T_base = self.compute_mobile_transform(x, y, z, roll, pitch, yaw)

        # Orbit and mount transform
        T_down = np.eye(4)
        T_down[:3, 3] = [0, 0, -0.19]

        T_orbit = np.eye(4)
        T_orbit[:3, :3] = R.from_euler('z', theta_orbit).as_matrix()

        T_mount = np.eye(4)
        T_mount[:3, 3] = [0.7, 0, 0]

        T_yflip = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ])

        T_orbital = T_down @ T_orbit @ T_mount @ T_yflip
        T_cumul = T_base @ T_orbital

        # === Build world-frame arm FK chain ===
        T_list = self.compute_arm_fk_all(q_arm)

        T_list_world = []
        for T_rel in T_list:
            T_cumul = T_cumul @ T_rel
            T_list_world.append(T_cumul)

        # Add optional tool correction if used in FK
        T_tcp = self.compute_dh_transform(0, 0, 0, np.pi / 2)
        T_ee = T_list_world[-1] @ T_tcp
        p_ee = T_ee[:3, 3]

        # === Initialize Jacobian ===
        J_v = np.zeros((3, n_total))
        J_w = np.zeros((3, n_total))

        # Mobile base: translational and rotational DoFs
        R_base = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        p_base = np.array([x, y, z])
        for j in range(3):
            a = np.eye(3)[:, j]  # world-frame unit axis for x, y, z
            J_v[:, j] = a  # translation axes
        for j in range(3):
            a = R_base @ np.eye(3)[:, j]  # rotated x/y/z axes
            J_w[:, 3 + j] = a
            J_v[:, 3 + j] = np.cross(a, p_ee - p_base)

        # Orbit joint (Z rotation at orbit link origin in world frame)
        T_orbit_world = T_base @ T_down  # origin of orbit joint
        z_axis = T_orbit[:3, 2]          # local Z
        z_axis_world = T_orbit_world[:3, :3] @ z_axis
        o_axis = T_orbit_world[:3, 3]    # position of orbit joint in world

        J_w[:, 6] = z_axis_world
        J_v[:, 6] = np.cross(z_axis_world, p_ee - o_axis)

        # Arm joints
        for i, T_joint in enumerate(T_list_world[:6]):

            z_i = T_joint[:3, 2]
            o_i = T_joint[:3, 3]
            idx = 7 + i
            J_w[:, idx] = z_i
            J_v[:, idx] = np.cross(z_i, p_ee - o_i)

        # Stack into full 6x13 Jacobian
        J = np.vstack((J_v, J_w))

        # === Optional FD Check ===
        delta = 1e-6
        dq = np.random.randn(n_total) * delta
        T1 = self.compute_total_forward_kinematics(q)
        T2 = self.compute_total_forward_kinematics(q + dq)

        pos1, pos2 = T1[:3, 3], T2[:3, 3]
        dpos = (pos2 - pos1) / delta

        R1 = R.from_matrix(T1[:3, :3])
        R2 = R.from_matrix(T2[:3, :3])
        drot = (R2 * R1.inv()).as_rotvec() / delta

        twist_fd = np.hstack((dpos, drot))
        twist_J = J @ dq

        # Compute error vectors
        error_linear = dpos - twist_J[:3]
        error_angular = drot - twist_J[3:]

        # Norms of each part
        #print("Jacobian FD error (total):    ", np.linalg.norm(twist_fd - twist_J))
        #print("Jacobian FD error (linear):   ", np.linalg.norm(error_linear))
        #print("Jacobian FD error (angular):  ", np.linalg.norm(error_angular))

        # Optional: print raw vectors
        #print("Δpos (FD):    ", dpos)
        #print("Δpos (Jac):   ", twist_J[:3])
        #print("Δrot (FD):    ", drot)
        #print("Δrot (Jac):   ", twist_J[3:])

        return J

 

    def mujoco_forward_kinematics(self, q, site_name="ee_site"):
        tmp = mujoco.MjData(self.model)

        # Body offset of the MP as written in the XML
        body_off = self.model.body('MP').pos          # [0, 0, 1.5]

        # -- translation joints (mp_joint_1–3)
        tmp.qpos[0] = q[0] - body_off[0]              # X slider
        tmp.qpos[1] = q[1] - body_off[1]              # Y slider
        tmp.qpos[2] = q[2] - body_off[2]              # Z slider  ← subtract 1.5 m

        # -- rotation joints (mp_joint_4–6)  
        #   XML gives the MP body a fixed quaternion (0 1 0 0) = 180° about Y.
        #   The hinge joints add *extra* roll/pitch/yaw on top of that, so we
        #   can use the world RPY values directly.
        tmp.qpos[3] = q[3]      # roll  (mp_joint_4)
        tmp.qpos[4] = q[4]      # pitch (mp_joint_5)
        tmp.qpos[5] = q[5]      # yaw   (mp_joint_6)


        #orbital joint
        tmp.qpos[6] = q[6]
        # -- UR-10 joints
        tmp.qpos[7:] = q[7:]     # shoulder_pan … wrist_3

        mujoco.mj_forward(self.model, tmp)

        sid   = self.model.site(site_name).id
        T     = np.eye(4)
        T[:3, 3]  = tmp.site_xpos[sid]
        T[:3, :3] = np.array(tmp.site_xmat[sid]).reshape(3, 3)
        return T





    def mujoco_jacobian(self, model, data, q, site_name='ee_site'):
        tmp_data = mujoco.MjData(model)
        tmp_data.qpos[:] = q  # ✅ Now it's safe — temp copy
        mujoco.mj_forward(model, tmp_data)

        J_pos = np.zeros((3, model.nv))
        J_rot = np.zeros((3, model.nv))

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        mujoco.mj_jacSite(model, tmp_data, J_pos, J_rot, site_id)

        return np.vstack((J_pos, J_rot))


    
    def update_cmd(self):
 
        began = False

        if not began:
        
            qin = self.sim.get_full_q_vector()
            qin_base = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            #qin_arm = np.array([0.0, 0, -1, 0, 1, 0])
            qin_arm = qin[7:]
        
        
        # --- Hold if not running ---
        if not self.running_traj:

            #this is to send small velocity commands so that the robot remains in position before running the trajectory

            J_cdpr = CableJacobian(self.x_mes, self.model_dict)
            cable_vel = -J_cdpr @ (0.001*(qin_base-self.sim.get_full_q_vector()[:6]))
            #print(cable_vel)

            #qin_arm = qin[7:]
            self.vel_cmd[:8] = cable_vel
            self.vel_cmd[8] = 0
            self.vel_cmd[9:] = self.sim.hold_kuka_pose(qin_arm)


            return

        # --- Step 1: Current pose ---
        began = True
        q = self.sim.get_full_q_vector()

        fext = np.zeros(6)
        fext[2] = -60 * 9.81  # Z gravity

        success_T, tensions = bary_sol(q[:6], fext, self.model_dict) #this function checks if there is a valid tension distribution of the cables for a given pose of the mobile platform
        #print(success_T)
        if not success_T:
            #if an invalid tension distribution is detected the movement will stop
            print("Invalid tension")
            print(tensions)
            self.running_traj = False
            return
        
        #forward kinematics of the system        
        #T_current = self.mujoco_forward_kinematics(q)
        T_current = self.compute_total_forward_kinematics(q)
        pos_current = T_current[:3, 3]
        rot_current = T_current[:3, :3]

        # --- Step 2: Desired pose ---
        pos_target = self.x_targ[:3]
        r_target   = R.from_euler('xyz', self.x_targ[3:])      # target Rotation object
        r_current  = R.from_matrix(rot_current)                # current Rotation object

        # --- Step 3: Compute pose error ---
 
        quat_current = r_current.as_quat()
        quat_target = r_target.as_quat()

 
        r_err = R.from_quat(quat_target) * R.from_quat(quat_current).inv()
        ang_err = r_err.as_rotvec()





        

        pos_err = pos_target - pos_current
        pos_err[np.abs(pos_err) < 1e-3] = 0.0
        ang_err[np.abs(ang_err) < 1e-3] = 0.0
        pose_error = np.hstack((pos_err, ang_err))

        pos_err_norm = np.linalg.norm(pos_err)
        ang_err_norm = np.linalg.norm(ang_err)

        if self.running_traj:
            if self._ptp_start_time is None:
                self._ptp_start_time = self.sim.sim_data.time

            t_now = self.sim.sim_data.time - self._ptp_start_time


            #storing the error norm on a list to plot
            self.error_history.append([pos_err_norm, ang_err_norm])
            self.time_history.append(t_now)
        
        
        # right before you build the QP
        #print("  p-err :", pose_error[:3])   # m
        #print("  R-err :", pose_error[3:])   # rad (axis-angle)


        

        #print(f"[DEBUG] pos_err norm: {pos_err_norm:.4f}, ang_err norm: {ang_err_norm:.4f}")


        #print("")

        #convergence check
        if pos_err_norm < 0.0001 and ang_err_norm < 0.0001:
            #print("[INFO] ✅ Converged.")
            self.vel_cmd[:9] = 0.0
            self.vel_cmd[9:] = self.sim.hold_kuka_pose(q[7:])



            self._ik_converged = True
            return
        else:
            self._ik_converged = False


        
        # --- Step 4: Solve QP to compute delta_q ---
        J = self.geometric_jacobian(q)
        #J = self.mujoco_jacobian(self.model, self.data, q)
        cond = np.linalg.cond(J)
        #print(f"[Jacobian] Condition number: {cond:.2e}")
        #print("")

        #different values of the proportional gain tested
        #Kp = np.diag([1, 1, 1, 1, 1, 1])
        #Kp = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        Kp = np.diag([1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
        norm_error = np.linalg.norm(pose_error)
        gain_scale = np.clip(norm_error, 0.4, 1)
        Kp_scaled = gain_scale * Kp

        #Kd = np.diag([1, 1, 1, 1, 1, 1])

        desired_twist = Kp@pose_error 


        #normalize the desired velocity to make sure the end effector is able to reach the target, otherwise it would get too close but never fully converge
        twist_norm = np.linalg.norm(desired_twist)
        if twist_norm > 1e-6:  # avoid division by zero
            desired_twist_normalized = desired_twist / twist_norm
        else:
            desired_twist_normalized = desired_twist * 0  # or leave it unchanged

        #added weight to penaliwe movement in certain joints of the system, in this case the rotation components of the moving platform, can be tuned
        W_joint = np.diag([0, 0, 0, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0])

        H = J.T @ J +W_joint

        f = -J.T @ desired_twist_normalized

        delta_t = self.dt  
        cdpr_q_min = np.array([-1.5, -0.7,  0.5, -0, -0, -0])
        cdpr_q_max = np.array([ 1.5,  0.7,  2,  0,  0,  0])
        #ur10_q_min = np.array([-np.pi,-3*np.pi/4, -3*np.pi/4,         1, -np.pi, 1, -np.pi])
        #ur10_q_max = np.array([ np.pi, 3*np.pi/4,  np.pi/2,  3*np.pi/4,  np.pi,  3*np.pi/2,  np.pi])
        ur10_q_min = np.array([-np.pi,-np.pi, -3*np.pi/4,         1, -np.pi, 1, -np.pi])
        ur10_q_max = np.array([ np.pi, np.pi,  np.pi,  np.pi,  np.pi,  2*np.pi,  np.pi])


        q_min = np.concatenate((cdpr_q_min, ur10_q_min))
        q_max = np.concatenate((cdpr_q_max, ur10_q_max))


        # --- Joint velocity limits ---
        base_vel_min = np.array([-1, -1, -1, -0, -0, -0])
        orb_vel_min = np.array([-0.7])
        arm_vel_min = np.array([-1, -1, -1, -1, -1, -1])
        vel_min = np.concatenate((base_vel_min,orb_vel_min,arm_vel_min))

        base_vel_max = np.array([1, 1, 1, 0, 0, 0])
        orb_vel_max = np.array([0.7])
        arm_vel_max = np.array([1, 1, 1, 1, 1, 1])
        vel_max = np.concatenate((base_vel_max,orb_vel_max,arm_vel_max))


        # --- Convert joint position limits to velocity bounds ---
        #q = np.clip(q, q_min, q_max)
        q_pred = q + self.prev_vel_cmd * self.dt  # Lookahead prediction

        lb_pos = ((q_min - q_pred) / delta_t)
        ub_pos = ((q_max - q_pred) / delta_t)


        # --- Acceleration limits ---
        accel_min = np.array([-0.2] * 13)  # tweak per joint as needed
        accel_max = np.array([ 0.2] * 13)

        # --- Convert to velocity bounds ---
        #   accel = (delta_q - prev_vel) / dt  →  delta_q ∈ [prev + a_min*dt, prev + a_max*dt]
        accel_lb = self.prev_vel_cmd + accel_min * delta_t
        accel_ub = self.prev_vel_cmd + accel_max * delta_t



        # --- Final bounds: enforce BOTH position and velocity constraints ---
        lb = np.maximum(lb_pos, vel_min,accel_lb)
        ub = np.minimum(ub_pos, vel_max,accel_ub)

        

        


        lb = np.minimum(lb, ub)
        ub = np.maximum(lb, ub)

        
        #secondary objective to make sure that the system tries to stay in a centered position between the joint limits
        epsilon = 0.01
        P = np.eye(J.shape[1]) - np.linalg.pinv(J) @ J
        q0 = 0.5 * (q_min + q_max)           # preferred posture
        K = np.eye(len(q)) * 0.05             # simple gain matrix

        v_null = K @ (q0 - q)                # target null-space motion

        H += epsilon * (P.T @ P)
        f += -epsilon * (P.T @ P @ v_null)


        

        try:
            #delta_q = solve_qp(H, f, lb=lb, ub=ub, solver="cvxopt")
            delta_q = solve_qp(H, f, lb=lb, ub=ub, solver="osqp")
        except Exception as e:
            print("[QP] ❌ Exception:", e)
            self.vel_cmd[:] = 0.0
            return

        if delta_q is None:
            print("[QP] ❌ Solve failed.")
            self.vel_cmd[:] = 0.0
            return
        
        
        
        # --- QP solve ---
        joint_velocities = delta_q  # ← Already joint velocity, do NOT divide by delta_t
        # Clip arm joint velocities
        delta_q = np.clip(delta_q, -vel_min, vel_max)
        
        #the following commented section corresponds to an inverse kinematic solver using only the pseudoinverse; which is used to compare
        """""
        # --- Step 4 (REPLACEMENT): PSEUDOINVERSE IK (no QP) ---

        J = self.geometric_jacobian(q)  
        Kp = np.diag([1.5, 1.5, 1.5, 1.5, 1.5, 1.5])

        # Use the un-normalized twist so the gain Kp sets the scale
        desired_twist = Kp @ pose_error  # already computed above; keep as-is

        twist_norm = np.linalg.norm(desired_twist)
        if twist_norm > 1e-6:  # avoid division by zero
            desired_twist_normalized = desired_twist / twist_norm
        else:
            desired_twist_normalized = desired_twist * 0  # or leave it unchanged

        # Damped least-squares pseudoinverse: J^+ = J^T (J J^T + λ^2 I)^(-1)
        lam = 1e-3  # small damping; increase (e.g., 1e-2) if near-singular
        JJt = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJt + (lam**2) * np.eye(J.shape[0]))

        # Joint velocities from task twist
        joint_velocities = J_pinv @ desired_twist_normalized

        # --- Joint velocity limits ---
        base_vel_min = np.array([-1, -1, -1, -0, -0, -0])
        orb_vel_min = np.array([-0.7])
        arm_vel_min = np.array([-1, -1, -1, -1, -1, -1])
        vel_min = np.concatenate((base_vel_min,orb_vel_min,arm_vel_min))

        base_vel_max = np.array([1, 1, 1, 0, 0, 0])
        orb_vel_max = np.array([0.7])
        arm_vel_max = np.array([1, 1, 1, 1, 1, 1])
        vel_max = np.concatenate((base_vel_max,orb_vel_max,arm_vel_max))

        # Enforce simple velocity limits by clipping (keeps your arrays above)
        joint_velocities = np.maximum(joint_velocities, vel_min)
        joint_velocities = np.minimum(joint_velocities, vel_max)
        """""
        

        # --- CDPR and arm ---
        dq_base = joint_velocities[:6]
        dq_arm = joint_velocities[6:]


        J_cdpr = CableJacobian(self.x_mes, self.model_dict)
        #cond_cdpr = np.linalg.cond(J_cdpr)
        #print(f"[Jacobian] Condition number: {cond_cdpr:.2e}")
        cable_vel = -J_cdpr @ dq_base

        qin_base = q[:6]
        qin_arm = q[7:]

        # --- Combine command ---
        self.vel_cmd[:8] = cable_vel
        self.vel_cmd[8:] = dq_arm
        self.prev_vel_cmd = joint_velocities.copy()







    
    #function that executes the discretized trajectory when the option "disc" is chosen at the start, needs to be revised 
    def executeTrajectory(self, t, points, sim):
        self.sim = sim
        self.debug_mode = True
        self.synced = False
        self.running_traj = True
        self._debug_counter = 0

        import matplotlib.pyplot as plt
        error_list = []
        time_list = []

        tf = t[-1]
        interpN = int(5 * tf)

        # Add initial pose
        x0 = self.sim.get_kuka_ee_pose()#self.x_mes.copy()
        points = np.vstack((x0, points))
        t = np.insert(t, 0, 0.0).astype(float)

        success, traj = interpolate_trajectory2(t, points, interpN, self.model_dict)
        print("Traj generated")

        if not success:
            print("Traj out of bounds! op canceled")
            self.debug_mode = False
            self.running_traj = False
            return False

        print("------ Interpolated Trajectory ------")
        for i in range(traj.xInterp.shape[0]):
            print(f"t = {traj.tInterp[i][0]:.2f}, pos = {traj.xInterp[i, :3].round(3)}, ori = {traj.xInterp[i, 3:].round(3)}")
        print("-------------------------------------")

        # Set timing parameters
        start_time = sim.sim_data.time

        # Loop through trajectory and stream targets
        # New: Time-synced streaming loop
        i = 0
        n_interp = traj.xInterp.shape[0]
        start_time = sim.sim_data.time

        while i < n_interp:
            current_time = sim.sim_data.time - start_time

            # Advance i if time has progressed
            while i < n_interp - 1 and traj.tInterp[i][0] < current_time:
                i += 1

            Xtarg = traj.xInterp[i]
            Xptarg = traj.xpInterp[i]

            self.update_targ(Xtarg, Xptarg)



            # Log error for plotting
            pos_err = np.linalg.norm(self.x_mes[:3] - Xtarg[:3])
            r_current = R.from_euler('xyz', self.x_mes[3:], degrees=False)
            r_target = R.from_euler('xyz', Xtarg[3:], degrees=False)
            r_error = r_target * r_current.inv()
            ori_err = np.linalg.norm(r_error.as_rotvec())

            error_list.append(ori_err)
            time_list.append(sim.sim_data.time)

            sleep(self.dt)

        # Final hold to stabilize
        for _ in range(100):
            self.update_targ(traj.xInterp[-1], np.zeros(6))
            # Stabilize arm while base converges
            qin = sim.get_full_q_vector()
            qin_arm = qin[6:]
            self.vel_cmd[8:] = sim.hold_kuka_pose(qin_arm)

            sleep(self.dt)


        self.debug_mode = False
        self.running_traj = False
        return True
     
            