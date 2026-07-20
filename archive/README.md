# Archive

This directory contains dead/archived code from the original codebase.

## To archive (manual step needed — permission issue):

```bash
cd /home/z/research_contact_handeye/verification/Sim
sudo mkdir -p archive/Num2

# Archive dead Num2 files (~25 files)
sudo mv Num2/phase0_xu_solver.py Num2/deterministic_solver.py \
  Num2/edge_visual_servo.py Num2/edge_visual_servo_v2.py \
  Num2/slam_calib.py Num2/unsupervised_edge_calib.py \
  Num2/golden_pose_search.py Num2/debug_solver.py \
  Num2/prove_calib_method.py Num2/quick_proof.py \
  Num2/acquisition_sim.py Num2/test_*.py \
  Num2/diagnose_init.py Num2/verify_*.py Num2/analyze_*.py \
  Num2/realistic_mc_v3.py Num2/fov_geometry.py Num2/plane_calib.py \
  Num2/fanuc_tool.py archive/Num2/

# Delete Num2_common/ (100% duplicate of common/)
sudo rm -rf Num2_common/
```

## Why archived

- **Duplicated files**: fov_geometry.py, plane_calib.py, calib_solver.py were copied 2-3 times across directories. All functions now live in `handeye_sim/`.
- **Dead experiments**: ~20 Num2 files were throwaway tests/debug scripts. Kept for reference.
- **Num2_common/**: Byte-identical to common/. Deleted.
