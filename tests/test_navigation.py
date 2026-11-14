import numpy as np
import pytest

import RNNTrialStructures as rts
from RNNTrialStructures import structures


def test_configs_validate_early():
    with pytest.raises(ValueError):
        rts.ArenaConfig(preset="unknown")
    with pytest.raises(ValueError):
        rts.ArenaConfig(preset="open", pillar_textures=(1, 2, 3, 4))
    with pytest.raises(ValueError):
        rts.ArenaConfig(preset="legacy_four_pillar", wall_textures=(1, 2, 3, 4))
    with pytest.raises(ValueError):
        rts.ArenaConfig(preset="occluded", pillar_textures=(1, 2, -1, 4))
    with pytest.raises(ValueError):
        rts.NavigationConfig(min_n_steps=10, max_n_steps=9)
    with pytest.raises(ValueError):
        rts.NavigationConfig(min_n_steps=1, max_n_steps=2, inputs=("unknown",))
    with pytest.raises(ValueError):
        rts.NavigationConfig(min_n_steps=1, max_n_steps=2, outputs=("gaze", "gaze"))
    with pytest.raises(ValueError):
        rts.SeedConfig(train=1, validation=1, test=2)


def test_angular_preferences_are_unique():
    config = rts.NavigationConfig(min_n_steps=2, max_n_steps=3)
    preferences = structures._angular_preferences(config)
    angles = np.asarray(preferences.μ)
    assert angles.shape == (16,)
    assert np.unique(np.round(angles, 6)).size == 16
    assert not np.isclose(angles[0], angles[-1])


@pytest.mark.parametrize(
    ("preset", "expected_obstacles"),
    [("open", 0), ("occluded", 4), ("legacy_four_pillar", 4)],
)
def test_arena_presets(preset, expected_obstacles):
    arena = rts.make_arena(rts.ArenaConfig(preset=preset))
    assert int(structures.jl.RNNTrialStructures.get_num_obstacles(arena)) == expected_obstacles


def _small_generator(seed):
    trial = rts.get_navigation_trialstruct(
        3,
        5,
        ["distance", "movement", "texture"],
        ["conjunction"],
    )
    return rts.get_batch_generator(
        trial,
        4,
        trajectory_seed=seed,
        binsize=2.0,
        binsize_wall=5.0,
        named=True,
    )


def test_generators_are_reproducible_and_independent():
    first = _small_generator(1234)()
    second = _small_generator(1234)()
    different = _small_generator(1235)()

    np.testing.assert_array_equal(first.inputs, second.inputs)
    np.testing.assert_array_equal(first.targets, second.targets)
    np.testing.assert_array_equal(first.valid_mask, second.valid_mask)
    assert not np.array_equal(first.inputs, different.inputs)


def test_named_batch_shapes_align():
    batch = _small_generator(7)()
    assert batch.inputs.ndim == batch.targets.ndim == batch.valid_mask.ndim == 3
    assert batch.inputs.shape[1:] == batch.targets.shape[1:]
    assert batch.targets.shape == batch.valid_mask.shape
    assert batch.place_bin.shape == batch.surface_bin.shape == batch.joint_bin.shape
    assert batch.place_bin.shape == batch.inputs.shape[1:]
    assert batch.position_xy.shape[1:] == batch.gaze_xy.shape[1:]
    assert batch.position_xy.shape[1:] == batch.inputs.shape[1:]
    assert batch.lengths.shape == (batch.inputs.shape[2],)


def test_batch_parameters_validate_early():
    trial = rts.get_navigation_trialstruct(3, 5, ["distance"], ["position"])
    with pytest.raises(ValueError):
        rts.get_batch_generator(trial, 0)
    with pytest.raises(ValueError):
        rts.get_batch_generator(trial, 1, p_hd=1.1)
    with pytest.raises(ValueError):
        rts.get_batch_generator(trial, 1, trajectory_seed=-1)
