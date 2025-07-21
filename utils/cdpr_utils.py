import numpy as np
from numpy.linalg import inv,cond,norm,pinv
from numpy import cos,sin
from numpy import arctan2 as atan2

from scipy.linalg import null_space
from scipy.spatial.transform import Rotation as R

np.set_printoptions(edgeitems=30, linewidth=100000)

def get_vertecis(N,tp,tmin,tmax):
    mcable = N.shape[0]
    a1,a2,a31,a32 = np.zeros((mcable,)),np.zeros((mcable,)),np.zeros((mcable,)),np.zeros((mcable,))

    for i in range(mcable):
        a1[i] = N[i,0]
        a2[i] = N[i,1]
        a31[i] = tmax-tp[i]
        a32[i] = tmin - tp[i]

    Iv = []
    appakMth = 1e-10

    for i in range(mcable-1):
        for j in range(i+1,mcable):
            M = np.array([ [a1[i],a2[i]] , [a1[j],a2[j]] ])
            cvpp = np.array([ [a31[i]] , [a31[j]] ])
            cvpm = np.array([ [a31[i]] , [a32[j]] ])
            cvmp = np.array([ [a32[i]] , [a31[j]] ])
            cvmm = np.array([ [a32[i]] , [a32[j]] ])

            cnd = cond(M,2)

            if(1/cnd > appakMth):
                Ipp = inv(M)@cvpp
                Ipm = inv(M)@cvpm
                Imp = inv(M)@cvmp
                Imm = inv(M)@cvmm
                Iv.append(Ipp)
                Iv.append(Ipm)
                Iv.append(Imp)
                Iv.append(Imm)
    
    return Iv

def filter_vertecis(Iv, N, tp, tmin, tmax):
    deltact = 1
    
    for i in range(len(Iv)-1,-1,-1):
        
        tsolv = tp+N@Iv[i].reshape((2,))
        
        if (np.min(tsolv) < tmin-deltact) or (np.max(tsolv) > tmax+deltact):  # if minimum smaller than tmin or maximum greater than tmax delete the element
            
            del Iv[i]

    return Iv
     
def discard_duplicate_vertecis(Iv):
    eps = 1e-6

    for i in range(len(Iv)-1,-1,-1):
        for j in range(0,i):

            dI = Iv[i]-Iv[j]
            if abs(dI[0]) < eps and abs(dI[1]) < eps:
                del Iv[i]
                break
    
    return Iv

def sort_vertecis(Iv):
    Ivmean = np.zeros((2,1))
    
    for i in range(len(Iv)):
        Ivmean += Iv[i]
    
    Ivmean /= len(Iv)

    anglerayv = []

    for i in range(len(Iv)):
        rayv = Iv[i] - Ivmean
        anglerayv.append(atan2(rayv[1],rayv[0]))

    #Ivsort = [x for _,x in sorted(zip(anglerayv,Iv))]
    Ivsort = [np.array(x) for _, x in sorted(zip(anglerayv, [v.tolist() for v in Iv]))]


    return Ivsort

def vertecis_barycenter(Ivsort):

    Apoly = 0
    nverts = len(Ivsort)
    
    xve, yve = [],[]
    for i in range(nverts+1):  # adding first element at the end
        xve.append(Ivsort[i%nverts][0])
        yve.append(Ivsort[i%nverts][1])
    

    for i in range(nverts):
        Apoly += (xve[i]*yve[i+1] - xve[i+1]*yve[i])[0]
    Apoly /= 2

    xbary, ybary = 0, 0
    for i in range(nverts):
        xbary += ((xve[i]+xve[i+1])*(xve[i]*yve[i+1]-xve[i+1]*yve[i]))[0]
        ybary += ((yve[i]+yve[i+1])*(xve[i]*yve[i+1]-xve[i+1]*yve[i]))[0]
    
    xbary /= 6*(Apoly+1e-5)
    ybary /= 6*(Apoly+1e-5)

    return np.array([xbary,ybary])

def bary_sol(X,fext,model_dict):
    tmin,tmax = model_dict["tmin"],model_dict["tmax"]

    W = Wrench(X,model_dict)
    N = null_space(W)
    
    
    tp = -pinv(W)@fext
        
    Iv = get_vertecis(N,tp,tmin,tmax)
    
    if len(Iv) > 0:
        Iv = filter_vertecis(Iv,N,tp,tmin,tmax)
    
    if len(Iv) > 0:
        Iv = discard_duplicate_vertecis(Iv)

    if len(Iv) > 0:
        Iv = sort_vertecis(Iv)
    
    if len(Iv) > 0:
        Lbary = vertecis_barycenter(Iv)

        t = tp + N@Lbary
        succ = not np.isnan(np.sum(t))

        return succ, t
    
    
    return False,tp


def skew(v):

    M = np.array([[    0,-v[2], v[1]],
                  [ v[2],    0,-v[0]],
                  [-v[1], v[0],    0]])
    
    return M

def Rbe(Th):
    ps,th,ph = Th[0],Th[1],Th[2]

    R = np.array([[ cos(ph)*cos(th), cos(ph)*sin(th)*sin(ps) - sin(ph)*cos(ps), cos(ph)*sin(th)*cos(ps) + sin(ph)*sin(ps)],
                  [ sin(ph)*cos(th), sin(ph)*sin(th)*sin(ps) + cos(ph)*cos(ps), sin(ph)*sin(th)*cos(ps) - cos(ph)*sin(ps)],
                  [-sin(th),                                   cos(th)*sin(ps),                           cos(th)*cos(ps)]])
    
    return R

def get_points(X,i,Ei,Ai):
    x = X[:3]
    th = X[3:]
    R = Rbe(th)

    B = R@Ei[i,:]
    mp_anchor = x + B
    base_anchor = Ai[i,:]
    U = base_anchor-mp_anchor 
    li = norm(U)   
    U = U/norm(U)

    return U,B,li

def Wrench(X,model_dict):
    Ei,Ai = model_dict["ei_pos"],model_dict["ai_pos"]
    W = np.zeros((6,8))
    
    for i in range(W.shape[1]):
        U,B,li = get_points(X, i, Ei, Ai)

        W[:3,i] = U
        W[3:,i] = np.cross(B,U)
    
    return W


def IGM(x,model_dict):
    Ei,Ai = model_dict["ei_pos"],model_dict["ai_pos"]

    L = np.zeros((8,))    
    for i in range(8):
        U,B,li = get_points(x, i, Ei, Ai)

        L[i] = li
    
    return L


def dyn_matrices(x, xp,model_dict):
    # calculating the dynamic model matrices
    me = model_dict["mp_mass"]
    Ig = model_dict["Ig"]

    Sp = np.zeros((3,))

    W = Wrench(x,model_dict)
    R = Rbe(x[3:])
    MSp = me*R@Sp
    MSpx = skew(MSp)

    M = np.zeros((6,6))
    M[:3,:3] = me*np.eye(3)
    Ip = R@Ig@R.T - (1/me)*MSpx@MSpx
    M[3:,3:] = Ip
    M[:3,3:] = -MSpx
    M[3:,:3] = MSpx


    Wg = np.zeros((6,3))
    Wg[:3,:3] = me*np.eye(3)
    Wg[3:,:3] = MSpx
    wg = Wg@np.array([0,0,-9.81])


    we = np.zeros((6,))

    Cxp = np.zeros((6,))
    om = xp[3:]
    Cxp[:3] = skew(om)@skew(om)@MSp
    Cxp[3:] = skew(om)@Ip@om

    return M,Cxp,we,wg

def IDM_t(x, xp, xpp,model_dict):
    M,Cxp,we,wg = dyn_matrices(x,xp,model_dict)

    b = -M@xpp -Cxp + we + wg
    
    succ,tau = bary_sol(x,b,model_dict)


    return succ,tau


def CableJacobian(x, model_dict):
    Ei = model_dict["ei_pos"]
    Ai = model_dict["ai_pos"]

    A = np.zeros((8, 6))

    for i in range(8):
        u, B, _ = get_points(x, i, Ei, Ai)  # u: unit cable direction, B: platform-side anchor in world frame
        r = B - x[:3]                       # r is the lever arm from CoM to cable attachment point

        A[i, :3] = u                        # linear part (direction of force)
        A[i, 3:] = np.cross(B, u)          # angular part (torque = r x force)

    return A

import cvxpy as cp
import numpy as np

def solve_tension_safe_velocity_qp(t_now, J, twist_des, model_dict, dt=1/500):
    tmin = model_dict["tmin"]
    tmax = model_dict["tmax"]

    # Desired unconstrained cable velocity
    v_nominal = J @ twist_des  # shape (8,)

    # Stiffness model
    K_stiff = 100.0
    K = K_stiff * np.eye(8)

    # Define optimization variable
    v = cp.Variable(8)

    # Predict tension
    t_pred = t_now - dt * K @ v
    
    print("twist_des:", twist_des.round(3))
    print("v_nominal:", v_nominal.round(3))
    print("t_now:", t_now.round(2))
    print("predicted tension range:", (t_now - dt * K @ v_nominal).round(2),"\n")


    # Objective: stay close to v_nominal, but limit magnitude
    beta = 1e-3
    objective = cp.Minimize(cp.sum_squares(v - v_nominal) + beta * cp.sum_squares(v))

    constraints = [
        t_pred >= tmin,
        t_pred <= tmax
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    if v.value is not None:
        #v_out = np.clip(v.value, -1.0, 1.0)  # match actuator ctrlrange
        v_out = v.value
        return v_out
    else:
        print("[WARN] QP failed — returning zeros")
        return np.zeros(8)

