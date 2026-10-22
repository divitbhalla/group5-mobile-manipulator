# Group 5 · MuJoCo Mobile Manipulator: Safe Stopping & Gripping

Emerging Technologies (701ME0E002) capstone · B.Tech Integrated Mechanical Sem VII · MPSTME, SVKM's NMIMS Mumbai
Team: Divit Bhalla (Group Lead · arm & contact physics) · Viraj Mote (chassis kinematics & MJCF) · Sameer Joglekar (perception & FMEA)

A warehouse robot drives at up to 0.3 m/s, must stop within 100 mm when its ultrasonic sensor sees an obstacle,
and must lift a 0.5 kg box without it slipping. This extends our M2 MuJoCo model (same chassis, wheels, arm and tote).

## Run it

```
pip install -r requirements.txt
python simulate_g5.py                          # live viewer: stop test, then grip test (Mac: mjpython)
python simulate_g5.py --test stop --floor-mu 0.1   # oily floor
python simulate_g5.py --test stop --snap 2.5       # ultrasonic wire snaps -> fail-safe stop
python simulate_g5.py --test grip --mu 0.5         # low friction: 4 N squeeze is not enough
python simulate_g5.py --test grip --mu 0.5 --squeeze 6
python simulate_g5.py --batch                  # no viewer: all tests, saves results/
```

## Hand calculation vs simulation

| Quantity | Hand calc | Sim | Difference |
|---|---|---|---|
| Stopping distance, dry floor, 0.3 m/s | s = 0.3×0.1 + 0.3²/(2×1.5) = 60 mm | 52.7 mm | −12 % (sim brakes at ≈ 2.0 m/s²) |
| Stopping distance, oily floor μ = 0.1 | 30 + 0.09/(2×0.98) = 76 mm | 75.1 mm | −1 % |
| Stop after sensor wire snaps | 0.3×(0.06+0.1) + 30 = 78 mm | 70.5 mm | −10 % |
| Min squeeze per finger, μ = 1.0 | N ≥ mg/(2μ) = 2.45 N | 2.62 N | +7 % |
| Min squeeze per finger, μ = 0.5 | 4.91 N | 5.22 N | +6 % |

Safety logic: NORMAL → WARNING (d ≤ 0.50 m) → SHUTDOWN (d ≤ 0.25 m, or no echo for 0.06 s), latched.
Formulas: s = v·t + v²/(2a) · v = ω·r · 2μN ≥ mg · d = c·t/2 · a_max = μg.
