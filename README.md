# RNNTrialStructures.py

Python interface to the Julia navigation-trial generator.

## Install

```bash
pip install git+https://github.com/SIONMATHEW/RNNTrialStructures.py.git
```

The Python package pins the Julia implementation to an immutable revision.

## Explicit configuration

```python
import RNNTrialStructures as rts

config = rts.NavigationConfig(
    min_n_steps=80,
    max_n_steps=120,
    inputs=("distance", "movement", "texture"),
    outputs=("conjunction",),
    arena=rts.ArenaConfig(preset="legacy_four_pillar"),
)

trial = rts.make_navigation_trialstruct(config)
generator = rts.get_batch_generator(
    trial,
    256,
    trajectory_seed=10_001,
    binsize=2.0,
    binsize_wall=5.0,
    named=True,
)
batch = generator()
```

`batch.inputs`, `batch.targets`, and `batch.valid_mask` retain Julia's
`(feature, time, trial)` axis order. Compact `place_bin`, `surface_bin`, and
`joint_bin` arrays use `(time, trial)`; continuous `position_xy` and `gaze_xy`
use `(coordinate, time, trial)`. The mask is zero on padded timesteps.

Three tested presets are available:

- `legacy_four_pillar`: original four-pillar experiment.
- `open`: four textured outer walls and no pillars.
- `occluded`: the same outer walls plus four separately textured pillars.

Use a different `trajectory_seed` namespace for training, validation, and
testing. `SeedConfig` provides non-overlapping defaults.

## Backward-compatible API

```python
trial = rts.get_navigation_trialstruct(
    20,
    50,
    ["distance", "movement", "texture"],
    ["conjunction"],
)
raw_generator = rts.get_batch_generator(trial, 500)
x, y, mask = raw_generator()
```

Calls without an arena preset continue to use `legacy_four_pillar`.
