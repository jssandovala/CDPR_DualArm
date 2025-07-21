import numpy as np
from time import perf_counter
from utils.cdpr_utils import *


class Trajectory:
    
    def __init__(self,tInterp,xInterp,xpInterp):
        self.tInterp = tInterp
        self.xInterp = xInterp
        self.xpInterp = xpInterp
        

def interpolate_linear(X,Y,x):
    n = X.shape[0]
    i = 0

    if x <= X[0]:
        return Y[0]
    elif x >= X[n-1]:
        return Y[n-1]
    else:
        while x > X[i+1]:
            i+=1
        
        xL,yL,xR,yR = X[i],Y[i],X[i+1],Y[i+1]

        dydx = (yR-yL)/(xR-xL)

        return yL + dydx*(x-xL)

def point_derivative_cubic(X, Y, i):

  if i == 0 or i == X.shape[0] - 1:
    return 0

  return (Y[i + 1] - Y[i - 1]) / (X[i + 1] - X[i - 1])

def interpolate_min_jerk(X, Y, x):
    n = X.shape[0]
    i = 0  # find left end of interval for interpolation

    if x <= X[0] :
        return [Y[0],0,0]

    elif x >= X[n-1]:
        return [Y[n-1],0,0]

    else :
        while x > X[i + 1]:
            i+=1

        xL, yL, xR, yR = X[i], Y[i], X[i+1], Y[i+1]
        t = x-xL
        d = xR-xL
        y = yL + (yR-yL)*(10*pow(t/d,3) - 15*pow(t/d,4) + 6*pow(t/d,5))
        yp = (yR-yL)*(30*pow(t/d,2)/d - 60*pow(t/d,3)/d + 30*pow(t/d,4)/d)
        ypp = (yR-yL)*(60*t/pow(d,3) - 180*pow(t/d,2)/pow(d,2) + 120*pow(t/d,3)/pow(d,2))

        return [y,yp,ypp]

def interpolate_trajectory(t, points, nInterp,model_dict):
    n = t.shape[0]
    tf = t[n-1]
    dt = tf / (nInterp - 1)
    succ = True

    tInterp = np.zeros((nInterp,1))
    xInterp = np.zeros((nInterp,6))
    xpInterp = np.zeros((nInterp,6))

    for i in range(nInterp):
        ct = i * dt
        tInterp[i] = ct

        
        for j in range(6):
            
            Y = interpolate_min_jerk(t, points[:,j], ct)
            xInterp[i, j] = Y[0]
            xpInterp[i, j] = Y[1]
    

    return succ, Trajectory(tInterp,xInterp,xpInterp)

from scipy.spatial.transform import Rotation as R, Slerp

def interpolate_trajectory2(t, points, nInterp, model_dict):
    print(">>> Waypoints:")
    for i in range(len(t)):
        print(f"t = {t[i]:.2f}, pos = {points[i, :3].round(3)}, ori = {points[i, 3:].round(3)}")

    succ = True

    # Split position and orientation (Euler)
    pos_points = points[:, :3]
    ori_points = points[:, 3:]  # Euler angles
    rot_keyframes = R.from_euler('xyz', ori_points)

    # Use unique time values for SLERP to avoid issues
    t_unique, indices = np.unique(t, return_index=True)
    t_slerp = t_unique
    rot_keyframes = rot_keyframes[indices]

    slerp = Slerp(t_slerp, rot_keyframes)

    tf = t[-1]

    # === Ensure interpolated time vector includes all key times ===
    # Uniform spacing for the rest
    dt = tf / (nInterp - len(t))
 
    t_interp_full = np.linspace(0, tf, nInterp)

    nInterp = len(t_interp_full)  # update in case it changed

    tInterp = np.zeros((nInterp, 1))
    xInterp = np.zeros((nInterp, 6))
    xpInterp = np.zeros((nInterp, 6))

    prev_rot = None

    for i in range(nInterp):
        ct = t_interp_full[i]
        tInterp[i] = ct

        # Position interpolation (min-jerk)
        for j in range(3):
            Y = interpolate_min_jerk(t, pos_points[:, j], ct)
            xInterp[i, j] = Y[0]
            xpInterp[i, j] = Y[1]

        # Orientation via SLERP
        rot_ct = slerp(ct)
        xInterp[i, 3:] = rot_ct.as_euler('xyz')

        # Approximate angular velocity
        if prev_rot is not None:
            delta_rot = rot_ct * prev_rot.inv()
            rotvec = delta_rot.as_rotvec()
            xpInterp[i, 3:] = rotvec / dt
        else:
            xpInterp[i, 3:] = np.zeros(3)

        prev_rot = rot_ct
    from utils.cdpr_utils import bary_sol

    fext = np.zeros(6)
    fext[2] = -50 * 9.81  # Z gravity

    xInterp_new = []
    xpInterp_new = []
    tInterp_new = []

    for i in range(len(xInterp)):
        
        pose = xInterp[i]
        xInterp_new.append(pose)
        xpInterp_new.append(xpInterp[i])
        tInterp_new.append(tInterp[i])

    # Recast to arrays
    xInterp = np.array(xInterp_new)
    xpInterp = np.array(xpInterp_new)
    tInterp = np.array(tInterp_new).reshape(-1, 1)




    return succ, Trajectory(tInterp, xInterp, xpInterp)



