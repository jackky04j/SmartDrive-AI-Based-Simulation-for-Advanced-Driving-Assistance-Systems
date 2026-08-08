# Local data and models

This directory has a stable layout so paths work after cloning the repository.

- `samples/` — local recorded driving videos. These are ignored because of repository size.
- `models/cone_detection/` — Phase 1 YOLO weights. `best.pt` is the default model.
- `models/drowsiness/` — place `shape_predictor_68_face_landmarks.dat` here for the Phase 2 driver monitor. The file is intentionally ignored: it is roughly 100 MB and exceeds GitHub's normal file limit.
- `models/*.obj` — CARLA cone assets.

The supplied commands fail with a clear path error if required local inputs are absent. You can pass alternate files with `--video` and `--model`.
