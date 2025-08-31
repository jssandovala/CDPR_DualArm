# CDPR_DualArm

A MuJoCo-based simulator of a Cable-Driven Parallel Robot (CDPR) with a dual-arm manipulator mounted on its moving platform

To run the simulation, execute `main.py` in the terminal. 

Once it launches, an input will appear in the terminal. Type `traj` to begin trajectory execution. 

It will then ask for a type: `ptp` runs a direct point-to-point trajectory, while `disc` discretizes and interpolates between the points. The main control logic is in `Controller.py`, in the `update_cmd()` function, which handles QP-based inverse kinematics and velocity command generation. 

The simulation loop is handled by the `Sim` class in `Sim.py`, which launches the viewer, collects sensor data, steps the simulation, and passes information to the controller. In this same file the initial pose of the arm is defined in init_sim.

General notes on adding a gripper to the end effector
------------------------------------------------

The current model ends at the tool frame, the ee_site on the file ur10_arm.xml . To integrate a gripper,
the following generic steps apply (independent of gripper type):


- Define a new body attached to the tool frame as the gripper base, on the ur10_arm.xml file.
- Add the geometry/joints required for the chosen gripper mechanism
    (e.g. hinge joints (revolute joints)for jaws, slide joint (prismatic joints)for a linear slider,
    etc.). A damping could be added in the joint configuration, which will act as viscuous friction. This parameter can be tuned
- Include corresponding actuators (position, velocity, or torque)
    to drive these joints in the file cdpr.xml. When setting a velocity actuator, the main components to tune there is kv, which is tied to the responsiveness of the joint.

