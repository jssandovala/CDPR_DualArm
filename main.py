from utils.model_params import *

from Sim import *

import threading
from time import sleep
from Controller import *
from scipy.spatial.transform import Rotation as R

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def run_point_to_point_test(sim, controller, points, hold_time=0.1):

    controller.error_history = []
    controller.time_history = []
    controller._ptp_start_time = None
    

    for i, pt in enumerate(points):
        print(f"\n[Target {i}] Going to:", pt)

        controller.update_targ(pt, np.zeros(6))
        controller.running_traj = True
        controller._debug_counter = 0
        

        reached = False
        t_start = sim.sim_data.time

        while not reached:
            sleep(0.01)

            q = sim.get_full_q_vector()
            T_now = controller.mujoco_forward_kinematics(q)
            pos_now = T_now[:3, 3]
            rot_now = T_now[:3, :3]

            pos_target = pt[:3]
            rot_target = R.from_euler('xyz', pt[3:]).as_matrix()

            pos_err = np.linalg.norm(pos_target - pos_now)
            rot_err = R.from_matrix(rot_target @ rot_now.T)
            ori_err = np.linalg.norm(rot_err.as_rotvec())
            


            if controller._ik_converged:
                print("reached")
                reached = True
                reached_time = sim.sim_data.time
                print("moving towards: ",points)

                
                
                

        # Hold for stability
        if reached:
            while sim.sim_data.time - reached_time < hold_time:
                controller.vel_cmd[:] = 0.0

                
                sleep(0.01)

        controller.running_traj = False
        print(f"[Reached] Target {i}")

        

        if i == 0:
            

            errors = np.array(controller.error_history)
            times = np.array(controller.time_history)

            if errors.ndim == 2 and len(times) == len(errors):
                plt.figure()
                plt.plot(times, errors[:, 0], label="Position Error (m)")
                plt.plot(times, errors[:, 1], label="Orientation Error (rad)")
                plt.xlabel("Time [s]")
                plt.ylabel("Error")
                plt.title("Pose Error Over Time (First PTP Target)")
                plt.grid(True)
                plt.legend()
                plt.savefig("pose_error_ptp_0.png")
                print("[INFO] Plot saved to pose_error_ptp_0.png")



        
        controller._ik_converged = False


def commands(sim:Sim):
    np.set_printoptions(edgeitems=30, linewidth=100000)

    cont = True
    sleep(3)
    while cont and sim.sim_open:
        cmd = input("command : ")

        if cmd == "exit" or cmd == "close" or cmd == "e":
            cont = False
            sim.close()

        elif cmd == "apply force":
            val = input("force vect (N N N Nm Nm Nm) : ")
            try:
                vect = np.array([float(el) for el in val.split(' ')])
                if vect.shape[0] != 6:
                    raise

                sim.applied_wrench = vect
                    
            except:
                print("invalid format")
        
        elif cmd == "traj":
            t = np.array([12,24,36])
            pts = np.array([#[1.738,-0.250,0.730,-1.571,0,-2.142],
                            [0,0,0.5,0,np.pi,0],
                            [1,0,0.5,0,np.pi,0],
                            [0,1,0.5,0,np.pi,0],
                            #[0,0,1.5,0,0,0],
                            #[0,1,1,0,0,0],
                            #[0,0,1,0,0,0]
                            ])

            trajectory_type = input("type : ")
            if trajectory_type == "disc":
                sim.controller.is_ptp = False
                sim.controller.executeTrajectory(t,pts,sim)
            elif trajectory_type == "ptp":
                sim.controller.is_ptp = True
                run_point_to_point_test(sim, sim.controller, pts)
        

        print()



sim = Sim(model_dict)

t1 = threading.Thread(target=commands, args=(sim,))
t1.start()

sim.launch_sim()

