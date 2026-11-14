"""Small Python interface to the Julia navigation trial generator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Literal, Sequence

import juliacall
import numpy as np


jl = juliacall.newmodule("RNNTrialStructs")
jl.seval("using RNNTrialStructures")
jl.seval("using Random")

ArenaPreset = Literal["legacy_four_pillar", "open", "occluded"]
_ARENA_PRESETS = {"legacy_four_pillar", "open", "occluded"}
_VALID_INPUTS = {
    "view", "head_direction", "movement", "position", "distance", "texture", "gaze"
}
_VALID_OUTPUTS = {
    "position", "head_direction", "view", "movement", "distance", "texture",
    "gaze", "conjunction"
}


@dataclass(frozen=True)
class ArenaConfig:
    """Geometry and surface identities for a navigation arena."""

    preset: ArenaPreset = "legacy_four_pillar"
    wall_textures: tuple[float, float, float, float] | None = None
    pillar_textures: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.preset not in _ARENA_PRESETS:
            raise ValueError(
                f"Unknown arena preset {self.preset!r}; expected one of "
                f"{sorted(_ARENA_PRESETS)}"
            )
        if self.preset == "legacy_four_pillar" and (
            self.wall_textures is not None or self.pillar_textures is not None
        ):
            raise ValueError("legacy_four_pillar has fixed legacy textures")
        if self.wall_textures is not None and len(self.wall_textures) != 4:
            raise ValueError("wall_textures must contain one value per outer wall")
        if self.preset == "open" and self.pillar_textures is not None:
            raise ValueError("open arenas do not contain pillars")
        if self.pillar_textures is not None and len(self.pillar_textures) != 4:
            raise ValueError("pillar_textures must contain one value per pillar")
        values = (self.wall_textures or ()) + (self.pillar_textures or ())
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("texture values must be finite and non-negative")


@dataclass(frozen=True)
class NavigationConfig:
    """Complete, serializable configuration for one navigation task."""

    min_n_steps: int
    max_n_steps: int
    inputs: tuple[str, ...] = ("distance", "movement", "texture")
    outputs: tuple[str, ...] = ("conjunction",)
    arena: ArenaConfig = ArenaConfig()
    angular_units: int = 16
    angular_concentration: float = 5.0
    angular_amplitude: float = 0.8

    def __post_init__(self) -> None:
        if self.min_n_steps < 1 or self.max_n_steps < self.min_n_steps:
            raise ValueError("step bounds must satisfy 1 <= min_n_steps <= max_n_steps")
        if self.angular_units < 2:
            raise ValueError("angular_units must be at least 2")
        if not math.isfinite(self.angular_concentration) or self.angular_concentration <= 0:
            raise ValueError("angular_concentration must be finite and positive")
        if not math.isfinite(self.angular_amplitude) or self.angular_amplitude <= 0:
            raise ValueError("angular_amplitude must be finite and positive")
        if not self.inputs:
            raise ValueError("at least one input is required")
        if not self.outputs:
            raise ValueError("at least one output is required")
        unknown_inputs = set(self.inputs) - _VALID_INPUTS
        unknown_outputs = set(self.outputs) - _VALID_OUTPUTS
        if unknown_inputs:
            raise ValueError(f"unknown navigation inputs: {sorted(unknown_inputs)}")
        if unknown_outputs:
            raise ValueError(f"unknown navigation outputs: {sorted(unknown_outputs)}")
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("navigation inputs must be unique")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("navigation outputs must be unique")


@dataclass(frozen=True)
class SeedConfig:
    """Independent seed namespaces for the three experimental splits."""

    train: int = 10_000
    validation: int = 1_000_000
    test: int = 2_000_000

    def __post_init__(self) -> None:
        seeds = (int(self.train), int(self.validation), int(self.test))
        if any(seed < 0 for seed in seeds):
            raise ValueError("split seeds must be non-negative")
        if len(set(seeds)) != 3:
            raise ValueError("training, validation, and test seeds must be distinct")

    def for_split(self, split: Literal["train", "validation", "test"]) -> int:
        return int(getattr(self, split))


@dataclass(frozen=True)
class NavigationBatch:
    """Python-owned arrays and compact categorical navigation labels."""

    inputs: np.ndarray
    targets: np.ndarray
    valid_mask: np.ndarray
    place_bin: np.ndarray
    surface_bin: np.ndarray
    joint_bin: np.ndarray
    position_xy: np.ndarray
    gaze_xy: np.ndarray
    lengths: np.ndarray


class NamedBatchGenerator:
    """Convert the legacy Julia tuple into named NumPy arrays."""

    def __init__(self, raw_generator):
        self.raw_generator = raw_generator

    def __call__(self) -> NavigationBatch:
        inputs, targets, valid_mask, labels = self.raw_generator()
        return NavigationBatch(
            inputs=np.asarray(inputs),
            targets=np.asarray(targets),
            valid_mask=np.asarray(valid_mask),
            place_bin=np.asarray(labels.place_bin),
            surface_bin=np.asarray(labels.surface_bin),
            joint_bin=np.asarray(labels.joint_bin),
            position_xy=np.asarray(labels.position_xy),
            gaze_xy=np.asarray(labels.gaze_xy),
            lengths=np.asarray(labels.lengths),
        )

    def __iter__(self):
        return self

    def __next__(self) -> NavigationBatch:
        return self()


def _julia_float_vector(values: Iterable[float]):
    return jl.Vector[jl.Float32]([jl.Float32(value) for value in values])


def _angular_preferences(config: NavigationConfig):
    # Julia ranges include both endpoints. Exclude 2pi so every direction is unique.
    angles = [
        2.0 * float(jl.pi) * index / config.angular_units
        for index in range(config.angular_units)
    ]
    return jl.RNNTrialStructures.AngularPreference(
        _julia_float_vector(angles),
        jl.Float32(config.angular_concentration),
        jl.Float32(config.angular_amplitude),
    )


def _symbol_vector(values: Sequence[str]):
    return jl.Vector[jl.Symbol]([jl.Symbol(value) for value in values])


def make_arena(config: ArenaConfig):
    """Construct one of the three versioned Julia arena presets."""

    kwargs = {}
    if config.wall_textures is not None:
        kwargs["wall_textures"] = _julia_float_vector(config.wall_textures)
    if config.pillar_textures is not None:
        kwargs["pillar_textures"] = _julia_float_vector(config.pillar_textures)
    return jl.RNNTrialStructures.navigation_arena(jl.Symbol(config.preset), **kwargs)


def make_navigation_trialstruct(config: NavigationConfig):
    """Create a Julia NavigationTrial from an explicit Python configuration."""

    return jl.RNNTrialStructures.NavigationTrial(
        int(config.min_n_steps),
        int(config.max_n_steps),
        _symbol_vector(config.inputs),
        _symbol_vector(config.outputs),
        make_arena(config.arena),
        _angular_preferences(config),
    )


def get_navigation_trialstruct(
    min_n_steps,
    max_n_steps,
    inputs,
    outputs,
    *,
    arena_type: ArenaPreset = "legacy_four_pillar",
    wall_textures=None,
    pillar_textures=None,
):
    """Backward-compatible convenience constructor."""

    arena = ArenaConfig(
        preset=arena_type,
        wall_textures=None if wall_textures is None else tuple(wall_textures),
        pillar_textures=None if pillar_textures is None else tuple(pillar_textures),
    )
    config = NavigationConfig(
        min_n_steps=int(min_n_steps),
        max_n_steps=int(max_n_steps),
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        arena=arena,
    )
    return make_navigation_trialstruct(config)


def get_batch_generator(
    trialstruct,
    ntrials,
    fov=jl.pi / 3,
    head_direction_step=jl.pi / 3,
    p_stay=1 / 3,
    p_hd=0.8,
    dt=20.0,
    binsize=1.0,
    binsize_wall=1.0,
    *,
    trajectory_seed=1,
    named=False,
):
    """Return an independently seeded Julia batch generator."""

    if int(ntrials) < 1:
        raise ValueError("ntrials must be positive")
    if float(dt) <= 0 or float(binsize) <= 0 or float(binsize_wall) <= 0:
        raise ValueError("dt and bin sizes must be positive")
    if not 0 <= float(p_stay) <= 1 or not 0 <= float(p_hd) <= 1:
        raise ValueError("p_stay and p_hd must be in [0, 1]")
    if not 0 < float(fov) <= 2 * math.pi:
        raise ValueError("fov must be in (0, 2pi]")
    if float(head_direction_step) <= 0:
        raise ValueError("head_direction_step must be positive")
    seed = int(trajectory_seed)
    if seed < 0:
        raise ValueError("trajectory_seed must be non-negative")
    rng = jl.Random.Xoshiro(seed)
    trial_generator = jl.RNNTrialStructures.generate_trials(
        trialstruct,
        jl.Int64(ntrials),
        dt,
        rng=rng,
        rseed=seed,
        fov=fov,
        hd_step=head_direction_step,
        p_stay=p_stay,
        p_hd=p_hd,
        binsize=binsize,
        binsize_wall=binsize_wall,
        return_labels=bool(named),
    )
    return NamedBatchGenerator(trial_generator) if named else trial_generator


def config_dict(config: NavigationConfig) -> dict:
    """Return a JSON-serializable experiment manifest fragment."""

    return asdict(config)


__all__ = [
    "ArenaConfig",
    "NavigationBatch",
    "NavigationConfig",
    "NamedBatchGenerator",
    "SeedConfig",
    "config_dict",
    "get_batch_generator",
    "get_navigation_trialstruct",
    "make_arena",
    "make_navigation_trialstruct",
]

