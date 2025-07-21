# CDPR_DualArm

A MuJoCo-based simulator of a Cable-Driven Parallel Robot (CDPR) with a dual-arm manipulator mounted on its moving platform

To run the simulation, execute `main.py` in the terminal. 

Once it launches, an input will appear in the terminal. Type `traj` to begin trajectory execution. 

It will then ask for a type: `ptp` runs a direct point-to-point trajectory, while `disc` discretizes and interpolates between the points. The main control logic is in `Controller.py`, in the `update_cmd()` function, which handles QP-based inverse kinematics and velocity command generation. 

The simulation loop is handled by the `Sim` class in `Sim.py`, which launches the viewer, collects sensor data, steps the simulation, and passes information to the controller.
