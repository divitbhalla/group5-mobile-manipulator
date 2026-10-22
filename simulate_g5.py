# =====================================================================
# ET Capstone | Group 5 | MuJoCo Mobile Manipulator: Safe Stopping & Gripping
# Team : Viraj Mote (A031) · Divit Bhalla (A032) · Sameer Joglekar (A039)
# Extends our M2 simulate.py (IntraLog-AGV: same chassis, wheels, arm, 0.5 kg tote)
#
# Added for the capstone:
#   1. Ultrasonic sensor (MuJoCo rangefinder) on the chassis front + a wall
#   2. Safety logic  NORMAL -> WARNING -> SHUTDOWN  with a 0.1 s reaction delay
#   3. A two-finger gripper on a lift above the pedestal that squeezes and lifts the tote
#   4. Telemetry: CSV logs, plots and a screenshot for the docket
#
# How to run
#   python simulate_g5.py                 -> live demo in the MuJoCo viewer (stop test, then grip test)
#   python simulate_g5.py --speed 0.2     -> change robot speed (m/s)
#   python simulate_g5.py --mu 0.5        -> change finger friction
#   python simulate_g5.py --test grip  -> only the grip test (also: --test stop)
#   python simulate_g5.py --snap 2.5  -> ultrasonic wire snaps at 2.5 s: fail-safe stop
#   python simulate_g5.py --batch         -> no viewer: runs every test, saves results/ (CSV, PNG, JSON)
#   (on a Mac use  mjpython simulate_g5.py  for the viewer)
# =====================================================================
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--speed", type=float, default=0.3, help="cruise speed v (m/s)")
ap.add_argument("--mu", type=float, default=1.0, help="finger/box friction coefficient")
ap.add_argument("--mass", type=float, default=0.5, help="box mass (kg)")
ap.add_argument("--squeeze", type=float, default=4.0, help="squeeze force per finger N (newton)")
ap.add_argument("--floor-mu", type=float, default=1.0, help="floor friction (oily floor = 0.1)")
ap.add_argument("--snap", type=float, default=None, help="ultrasonic wire snaps at this time (s): sensor fault test")
ap.add_argument("--test", choices=["stop", "grip", "both"], default="both", help="which test to run live")
ap.add_argument("--batch", action="store_true", help="run all tests without viewer and save results/")
args = ap.parse_args()

if args.batch and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

TEAM = "Group 5 · A031 A032 A039"

# ---------------- design numbers (same as hand calculation, Section 2) ----------------
R_WHEEL = 0.09     # m, wheel radius (from M2)
T_REACT = 0.10     # s, sensor + code reaction time
A_BRAKE = 1.5      # m/s^2, braking deceleration we command
A_UP    = 0.5      # m/s^2, gentle start
D_WARN  = 0.50     # m, ultrasonic distance -> WARNING
D_TRIP  = 0.25     # m, ultrasonic distance -> SHUTDOWN (brake)
FAULT_T = 0.06     # s, no echo for one sensor cycle -> sensor FAULT -> SHUTDOWN
WALL_X  = 1.60     # m, wall face position
LIFT_H  = 0.10     # m, how high the gripper lifts the box
SLIP_MM = 5.0      # mm, box sliding more than this inside the fingers = SLIP


def build_xml(mu=1.0, mass=0.5, floor_mu=1.0):
    return f"""
<mujoco model="G5-MobileManipulator">
  <compiler angle="degree"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.8 0.8 0.8" width="512" height="512"/>
    <material name="checker" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light name="sun" pos="0 0 5" dir="0 0 -1" directional="true"/>
    <light pos="1 -2 3" dir="-0.3 0.6 -1"/>
    <geom name="floor" type="plane" size="2.5 2.5 0.1" material="checker" friction="{floor_mu} 0.005 0.0001" priority="1"/>
    <!-- obstacle (rack / wall) in front of the robot -->
    <geom name="wall" type="box" pos="{WALL_X + 0.05} 0 0.3" size="0.05 0.8 0.3" rgba="0.8 0.2 0.2 1"/>
    <!-- pick station: pedestal from M2, moved to the side -->
    <geom name="pedestal" type="box" pos="0.3 1.1 0.25" size="0.2 0.2 0.25" rgba="0.6 0.4 0.2 1"/>
    <camera name="side_cam" pos="0.8 -2.6 1.3" xyaxes="1 0 0 0 0.45 0.9" fovy="55"/>
    <camera name="grip_cam" pos="-0.25 1.1 0.72" xyaxes="0 -1 0 0.3 0 1" fovy="35"/>

    <!-- M2 chassis, 6-DOF free joint -->
    <body name="chassis" pos="0 0 0.12">
      <freejoint name="chassis_free"/>
      <geom name="chassis_geom" type="box" size="0.25 0.18 0.06" mass="10" rgba="0.1 0.5 0.9 1"/>
      <site name="imu_site" pos="0 0 0" size="0.01"/>
      <!-- NEW: ultrasonic sensor on the front face, looking forward (+x) -->
      <geom name="us_box" type="box" pos="0.262 0 0.02" size="0.012 0.04 0.02" mass="0.01" rgba="0.2 0.2 0.2 1" contype="0" conaffinity="0"/>
      <site name="us_site" pos="0.275 0 0.02" zaxis="1 0 0" size="0.01" rgba="0 1 1 1"/>
      <!-- M2 2-DOF arm -->
      <body name="turret" pos="0 0 0.08">
        <joint name="arm_turret_yaw" type="hinge" axis="0 0 1" range="-90 90" damping="5" armature="0.05"/>
        <geom type="cylinder" size="0.05 0.02" mass="0.5" rgba="0.3 0.3 0.3 1"/>
        <body name="arm_link" pos="0 0 0.03">
          <joint name="arm_shoulder_pitch" type="hinge" axis="0 1 0" range="-60 60" damping="5" armature="0.1"/>
          <geom type="capsule" fromto="0 0 0 0.35 0 0" size="0.02" mass="0.3" rgba="0.9 0.6 0.1 1"/>
          <site name="ee_tip" pos="0.35 0 0" size="0.02" rgba="1 0 0 1"/>
        </body>
      </body>
      <!-- M2 wheels r = 0.09 m -->
      <body name="wheel_fl" pos="0.2 0.22 -0.03"><joint name="wheel_fl_joint" type="hinge" axis="0 1 0" damping="0.15" armature="0.01"/>
        <geom type="cylinder" size="0.09 0.025" zaxis="0 1 0" mass="1" rgba="0.1 0.1 0.1 1"/></body>
      <body name="wheel_fr" pos="0.2 -0.22 -0.03"><joint name="wheel_fr_joint" type="hinge" axis="0 1 0" damping="0.15" armature="0.01"/>
        <geom type="cylinder" size="0.09 0.025" zaxis="0 1 0" mass="1" rgba="0.1 0.1 0.1 1"/></body>
      <body name="wheel_rl" pos="-0.2 0.22 -0.03"><joint name="wheel_rl_joint" type="hinge" axis="0 1 0" damping="0.15" armature="0.01"/>
        <geom type="cylinder" size="0.09 0.025" zaxis="0 1 0" mass="1" rgba="0.1 0.1 0.1 1"/></body>
      <body name="wheel_rr" pos="-0.2 -0.22 -0.03"><joint name="wheel_rr_joint" type="hinge" axis="0 1 0" damping="0.15" armature="0.01"/>
        <geom type="cylinder" size="0.09 0.025" zaxis="0 1 0" mass="1" rgba="0.1 0.1 0.1 1"/></body>
    </body>

    <!-- M2 cargo tote (the box), sits on the pedestal -->
    <body name="cargo_tote" pos="0.3 1.1 0.52">
      <freejoint name="tote_free"/>
      <geom name="tote" type="box" size="0.05 0.05 0.02" mass="{mass}" friction="{mu} 0.005 0.0001" rgba="0 0.8 0.2 1"/>
    </body>

    <!-- NEW: two-finger gripper on a vertical lift above the pedestal -->
    <body name="lift" pos="0.3 1.1 0.52">
      <joint name="lift_z" type="slide" axis="0 0 1" range="-0.02 0.3" damping="20"/>
      <geom type="box" pos="0 0 0.09" size="0.03 0.1 0.01" mass="0.3" rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
      <body name="finger_L" pos="0 -0.085 0">
        <joint name="fL" type="slide" axis="0 1 0" range="0 0.05" damping="5"/>
        <geom type="box" pos="0 0 0.05" size="0.008 0.008 0.04" mass="0.02" rgba="0.4 0.4 0.4 1" contype="0" conaffinity="0"/>
        <geom name="pad_L" type="box" size="0.03 0.01 0.018" mass="0.05" friction="{mu} 0.005 0.0001" rgba="0.15 0.15 0.15 1"/>
        <site name="touch_L" pos="0 0.012 0" type="box" size="0.032 0.006 0.02" rgba="1 0 0 0.3"/>
      </body>
      <body name="finger_R" pos="0 0.085 0">
        <joint name="fR" type="slide" axis="0 -1 0" range="0 0.05" damping="5"/>
        <geom type="box" pos="0 0 0.05" size="0.008 0.008 0.04" mass="0.02" rgba="0.4 0.4 0.4 1" contype="0" conaffinity="0"/>
        <geom name="pad_R" type="box" size="0.03 0.01 0.018" mass="0.05" friction="{mu} 0.005 0.0001" rgba="0.15 0.15 0.15 1"/>
      </body>
    </body>
  </worldbody>

  <contact>
    <exclude body1="lift" body2="world"/>
    <exclude body1="finger_L" body2="world"/>
    <exclude body1="finger_R" body2="world"/>
  </contact>

  <actuator>
    <velocity name="v_fl" joint="wheel_fl_joint" kv="20" ctrlrange="-15 15"/>
    <velocity name="v_fr" joint="wheel_fr_joint" kv="20" ctrlrange="-15 15"/>
    <velocity name="v_rl" joint="wheel_rl_joint" kv="20" ctrlrange="-15 15"/>
    <velocity name="v_rr" joint="wheel_rr_joint" kv="20" ctrlrange="-15 15"/>
    <motor name="m_shoulder" joint="arm_shoulder_pitch" gear="12" ctrlrange="-25 25"/>
    <position name="lift" joint="lift_z" kp="3000" ctrlrange="-0.02 0.3"/>
    <!-- squeeze force per finger (newton). Stands in for the Nitinol wire pull -->
    <motor name="squeeze_L" joint="fL" ctrlrange="0 20"/>
    <motor name="squeeze_R" joint="fR" ctrlrange="0 20"/>
  </actuator>

  <sensor>
    <rangefinder name="ultrasonic" site="us_site"/>
    <jointvel name="wheel_tacho" joint="wheel_fl_joint"/>
    <accelerometer name="imu_acc" site="imu_site"/>
    <touch name="finger_touch" site="touch_L"/>
  </sensor>
</mujoco>
"""


class Sim:
    def __init__(self, mu=1.0, mass=0.5, floor_mu=1.0, viewer=None):
        self.m = mujoco.MjModel.from_xml_string(build_xml(mu, mass, floor_mu))
        self.d = mujoco.MjData(self.m)
        self.viewer = viewer
        self.pitch_q = self.m.jnt_qposadr[self.m.joint("arm_shoulder_pitch").id]
        self.pitch_v = self.m.jnt_dofadr[self.m.joint("arm_shoulder_pitch").id]
        self.lift_q = self.m.jnt_qposadr[self.m.joint("lift_z").id]
        self.tote_q = self.m.jnt_qposadr[self.m.joint("tote_free").id]
        mujoco.mj_forward(self.m, self.d)

    def hold_arm(self):
        # keep the arm level (simple PD) so it does not flop while driving
        q = self.d.qpos[self.pitch_q]; w = self.d.qvel[self.pitch_v]
        self.d.ctrl[4] = np.clip(-(40 * q + 4 * w), -25, 25)

    def step(self):
        mujoco.mj_step(self.m, self.d)
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
            time.sleep(self.m.opt.timestep)

    # ---------------- TEST A: drive, see obstacle, stop ----------------
    def stop_test(self, v_set=0.3, t_max=12.0, snap_at=None):
        m, d = self.m, self.d
        dt = m.opt.timestep
        v_cmd, state = 0.0, "NORMAL"
        t_trip = x_trip = t_stop = x_snap = None
        t_valid, cause = 0.0, None
        log = []
        while d.time < t_max:
            us = d.sensor("ultrasonic").data[0]          # metres, -1 if no echo
            if snap_at is not None and d.time >= snap_at:
                if x_snap is None: x_snap = d.qpos[0]
                us = -1.0                                  # wire snapped: no echo any more
            if us >= 0: t_valid = d.time
            if state != "SHUTDOWN":
                if us < 0 and d.time - t_valid >= FAULT_T:  # fail-safe: no echo = we are blind = stop
                    state, t_trip, x_trip, cause = "SHUTDOWN", d.time, d.qpos[0], "SENSOR FAULT"
                elif us >= 0:
                    state = "WARNING" if us <= D_WARN else "NORMAL"
                    if us <= D_TRIP:
                        state, t_trip, x_trip, cause = "SHUTDOWN", d.time, d.qpos[0], "OBSTACLE"
            if self.viewer is not None and log and ["NORMAL", "WARNING", "SHUTDOWN"].index(state) != log[-1][5]:
                why = f" ({cause})" if state == "SHUTDOWN" else ""
                print(f"  t = {d.time:5.2f} s | ultrasonic d = {us:.3f} m | state -> {state}{why}")
            # speed command
            if state != "SHUTDOWN" or d.time < t_trip + T_REACT:
                v_cmd = min(v_set, v_cmd + A_UP * dt)          # still driving (reaction time)
            else:
                v_cmd = max(0.0, v_cmd - A_BRAKE * dt)         # braking
            d.ctrl[0:4] = v_cmd / R_WHEEL                      # v = w * r  ->  w = v / r
            self.hold_arm()
            self.step()
            v = d.qvel[0]
            log.append((d.time, d.qpos[0], v, v_cmd, us, ["NORMAL", "WARNING", "SHUTDOWN"].index(state), d.sensor("wheel_tacho").data[0] * R_WHEEL))
            if state == "SHUTDOWN" and t_stop is None and d.time > t_trip + T_REACT + 0.05 and abs(v) < 0.001:
                t_stop = d.time
            if t_stop is not None and d.time > t_stop + 0.5:
                break
        log = np.array(log)
        gap = WALL_X - (d.qpos[0] + 0.275)
        res = dict(v_set=v_set, t_trip=t_trip, x_trip=x_trip, cause=cause,
                   from_snap_mm=(d.qpos[0] - x_snap) * 1000 if x_snap is not None else None,
                   stop_dist_mm=(d.qpos[0] - x_trip) * 1000 if x_trip is not None else None,
                   response_s=(t_stop - t_trip) if t_stop else None,
                   final_gap_mm=gap * 1000, max_speed=float(log[:, 2].max()))
        return res, log

    # ---------------- TEST B: squeeze, lift, hold the box ----------------
    def grip_test(self, squeeze=4.0, t_max=4.0):
        m, d = self.m, self.d
        log = []
        rel0 = None
        while d.time < t_max:
            t = d.time
            f = squeeze if t > 0.3 else 0.0                    # close fingers
            d.ctrl[6] = d.ctrl[7] = f
            if t < 1.0: z = 0.0
            elif t < 2.0: z = LIFT_H * (1 - np.cos(np.pi * (t - 1.0))) / 2   # lift 0.1 m in 1 s, smooth start/stop
            else: z = LIFT_H                                    # hold
            d.ctrl[5] = z
            self.hold_arm()
            self.step()
            lift_z = d.qpos[self.lift_q]; box_z = d.qpos[self.tote_q + 2] - 0.52
            if t >= 0.95 and rel0 is None: rel0 = box_z - lift_z
            slip = 0.0 if rel0 is None else (rel0 - (box_z - lift_z)) * 1000
            log.append((t, lift_z, box_z, slip, d.sensor("finger_touch").data[0]))
        log = np.array(log)
        hold = log[log[:, 0] > 2.5]
        res = dict(squeeze_N=squeeze, slip_mm=float(log[-1, 3]),
                   touch_N=float(hold[:, 4].mean()), box_lifted_mm=float(log[-1, 2] * 1000))
        res["result"] = "SLIP" if res["slip_mm"] > SLIP_MM else "HOLDS"
        return res, log


def snapshot(sim, cam, path, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    r = mujoco.Renderer(sim.m, 720, 1280)
    r.update_scene(sim.d, camera=cam); img = r.render(); r.close()
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100); ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis("off")
    ax.text(20, 40, f"{TEAM}  |  MuJoCo {mujoco.__version__}", color="w", fontsize=18, weight="bold",
            bbox=dict(facecolor="black", alpha=0.7, pad=6))
    ax.text(20, 690, title, color="w", fontsize=15, bbox=dict(facecolor="black", alpha=0.7, pad=5))
    fig.savefig(path); plt.close(fig)


def batch():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = "results"; os.makedirs(out, exist_ok=True)
    R = {"stop": [], "grip": []}

    # Test A: stop at 0.3 and 0.2 m/s, plus oily floor
    stop_runs = [(0.3, 1.0), (0.2, 1.0), (0.3, 0.1)]
    sf = Sim(); res, log = sf.stop_test(0.3, snap_at=2.5); res["floor_mu"] = 1.0; res["test"] = "sensor wire snaps at 2.5 s"
    R["stop"].append(res); print("FAULT", res)
    np.savetxt(f"{out}/stop_sensor_fault.csv", log, delimiter=",", fmt="%.5f",
               header="t_s,x_m,v_mps,v_cmd_mps,ultrasonic_m,state(0=N,1=W,2=S),wheel_v_mps")
    logs = {}
    for v, fmu in stop_runs:
        s = Sim(floor_mu=fmu); res, log = s.stop_test(v)
        res["floor_mu"] = fmu; R["stop"].append(res); logs[(v, fmu)] = log
        np.savetxt(f"{out}/stop_v{v}_floor{fmu}.csv", log, delimiter=",", fmt="%.5f",
                   header="t_s,x_m,v_mps,v_cmd_mps,ultrasonic_m,state(0=N,1=W,2=S),wheel_v_mps")
        if v == 0.3 and fmu == 1.0:
            snapshot(s, "side_cam", f"{out}/screenshot_stop.png",
                     f"Stopped: {res['stop_dist_mm']:.0f} mm after trip · gap to wall {res['final_gap_mm']:.0f} mm · state SHUTDOWN")
        print("STOP", res)

    # Test B: grip
    grip_runs = [(1.0, 0.5, 4.0), (0.5, 0.5, 4.0), (0.5, 0.5, 6.0), (1.0, 0.75, 4.0), (1.0, 0.5, 2.0)]
    glogs = {}
    for mu, mass, N in grip_runs:
        s = Sim(mu=mu, mass=mass); res, log = s.grip_test(N)
        res.update(mu=mu, mass_kg=mass, N_needed=mass * 9.81 / (2 * mu)); R["grip"].append(res); glogs[(mu, mass, N)] = log
        np.savetxt(f"{out}/grip_mu{mu}_m{mass}_N{N}.csv", log, delimiter=",", fmt="%.5f",
                   header="t_s,lift_z_m,box_z_m,slip_mm,touch_N")
        if (mu, mass, N) == (1.0, 0.5, 4.0):
            snapshot(s, "grip_cam", f"{out}/screenshot_grip.png", f"mu = 1.0, squeeze 4 N/finger: box HOLDS (slip {res['slip_mm']:.1f} mm)")
        if (mu, mass, N) == (0.5, 0.5, 4.0):
            snapshot(s, "grip_cam", f"{out}/screenshot_slip.png", f"mu = 0.5, squeeze 4 N/finger: box SLIPS ({res['slip_mm']:.0f} mm)")
        print("GRIP", res)

    # Plot: telemetry for Section 3
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    L = logs[(0.3, 1.0)]; r0 = R["stop"][0]
    ax[0].plot(L[:, 0], L[:, 2], label="robot speed v (m/s)", lw=2)
    ax[0].plot(L[:, 0], np.minimum(L[:, 4], 1.2) / 4, "--", label="ultrasonic d / 4 (m)")
    ax[0].axvline(r0["t_trip"], color="r", lw=1); ax[0].text(r0["t_trip"] + 0.05, 0.33, "trip d = 0.25 m", color="r")
    ax[0].axvline(r0["t_trip"] + T_REACT, color="k", lw=1, ls=":"); ax[0].text(r0["t_trip"] + T_REACT + 0.05, 0.29, "brake", color="k")
    ax[0].set_xlabel("time (s)"); ax[0].set_title(f"Stop test: v = 0.3 m/s -> stopped in {r0['stop_dist_mm']:.0f} mm")
    ax[0].set_xlim(max(0, r0["t_trip"] - 2), r0["t_trip"] + 1); ax[0].set_ylim(0, 0.36); ax[0].legend(loc="lower left"); ax[0].grid(alpha=.3)
    for key, lab in [((1.0, 0.5, 4.0), "mu 1.0, 4 N"), ((0.5, 0.5, 4.0), "mu 0.5, 4 N"), ((0.5, 0.5, 6.0), "mu 0.5, 6 N (fix)")]:
        G = glogs[key]; ax[1].plot(G[:, 0], G[:, 3], lw=2, label=lab)
    ax[1].axhline(SLIP_MM, color="r", ls="--", lw=1); ax[1].text(0.1, SLIP_MM + 1, "slip limit 5 mm", color="r")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("box slip inside fingers (mm)"); ax[1].set_ylim(-2, 40)
    ax[1].set_title("Grip test: 0.5 kg box lifted 0.1 m"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle(TEAM, fontsize=10, x=0.99, ha="right"); fig.tight_layout(); fig.savefig(f"{out}/telemetry.png", dpi=150)
    json.dump(R, open(f"{out}/results.json", "w"), indent=2)
    print("saved to", out)


def live():
    import mujoco.viewer
    s = Sim(mu=args.mu, mass=args.mass, floor_mu=args.floor_mu)
    with mujoco.viewer.launch_passive(s.m, s.d) as v:
        s.viewer = v
        if args.test in ("stop", "both"):
            v.cam.distance, v.cam.elevation, v.cam.azimuth, v.cam.lookat[:] = 3.2, -20, 90, (0.8, 0.3, 0.2)
            print(f"[{TEAM}] TEST A: stop test at v = {args.speed} m/s, floor mu = {args.floor_mu}")
            res, _ = s.stop_test(args.speed, snap_at=args.snap)
            if res["cause"] == "SENSOR FAULT":
                print(f"  sensor wire snapped at t = {args.snap} s -> robot stopped {res['from_snap_mm']:.1f} mm later | gap to wall = {res['final_gap_mm']:.0f} mm")
                res["stop_dist_mm"] = res["from_snap_mm"]
            else:
                print(f"  trip at d = {D_TRIP} m -> stopping distance = {res['stop_dist_mm']:.1f} mm | gap to wall = {res['final_gap_mm']:.0f} mm")
            print(f"  {'PASS' if res['stop_dist_mm'] <= 100 else 'FAIL'} (target <= 100 mm)")
        if args.test in ("grip", "both"):
            v.cam.distance, v.cam.elevation, v.cam.azimuth, v.cam.lookat[:] = 0.9, -10, 90, (0.3, 1.1, 0.55)
            print(f"[{TEAM}] TEST B: grip test mu = {args.mu}, box {args.mass} kg, squeeze {args.squeeze} N per finger")
            print(f"  needed N >= m*g/(2*mu) = {args.mass*9.81/(2*args.mu):.2f} N")
            s.d.time = 0.0   # restart the clock for the grip schedule
            res, _ = s.grip_test(args.squeeze)
            print(f"  slip = {res['slip_mm']:.1f} mm | touch sensor = {res['touch_N']:.2f} N -> {res['result']}")
        print("Close the viewer window to exit.")
        while v.is_running():
            v.sync(); time.sleep(0.02)


if __name__ == "__main__":
    batch() if args.batch else live()
