"""RobStride RS01 motor constants for the FanfanA1Clean training configs.

These values are intentionally plain constants so the robot, reward, and
training notes all reference the same source.  Values marked as engineering
initial values are not from the motor manual and should be identified on the
real robot later.
"""

# From RS01 manual: peak torque in motion-control mode is +/-17 N*m.
RS01_PEAK_TORQUE = 17.0

# From RS01 manual: rated continuous load torque is 6 N*m.
RS01_CONTINUOUS_TORQUE = 6.0

# From RS01 manual: no-load speed is about 315 rpm = 33 rad/s.
RS01_VELOCITY_LIMIT = 33.0

# From RS01 manual: rated-load speed is about 100 rpm = 10.47 rad/s.
RS01_RATED_VELOCITY = 10.47

# Real robot gains that currently run best for this platform.
RS01_KP = 40.0
RS01_KD = 5.0

# Safe first-stage policy action scale.  0.15 rad is about 8.6 degrees.
RS01_ACTION_SCALE_SAFE = 0.15

# Normal action scale to try after the safe version is stable.
# 0.20 rad is about 11.5 degrees.
RS01_ACTION_SCALE_NORMAL = 0.20

# Engineering initial value, not in the RS01 manual.  Needs hardware ID later.
RS01_ARMATURE = 0.01

# Engineering initial value, not in the RS01 manual.  Needs hardware ID later.
RS01_FRICTION = 0.08

# From RS01 manual: one motor is about 0.38 kg.
RS01_MOTOR_MASS = 0.38

# 12 RS01 motors add about 4.56 kg.  Check whether the URDF/USD includes this.
RS01_TOTAL_MOTOR_MASS_12DOF = 12.0 * RS01_MOTOR_MASS

# Randomization ranges.  These are conservative engineering ranges.
RS01_STIFFNESS_SCALE_RANGE = (0.8, 1.2)
RS01_DAMPING_SCALE_RANGE = (0.7, 1.3)
RS01_JOINT_FRICTION_RANGE = (0.03, 0.15)
RS01_ARMATURE_RANGE = (0.005, 0.02)

# Deployment-like target filter ranges.  6 N*m is the continuous RS01 rating;
# 17 N*m is only a short peak/hard cap and should not be treated as the normal
# walking budget.
RS01_DEPLOY_TORQUE_BUDGET_RANGE = (5.0, 10.0)
RS01_DEPLOY_SHORT_PEAK_TORQUE_RANGE = (10.0, 14.0)
RS01_MOTOR_STRENGTH_SCALE_RANGE = (0.65, 1.05)
