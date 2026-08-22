import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "hy3dworld" / "models" / "pipelines.py"
)
SPEC = importlib.util.spec_from_file_location("hunyuan_pipelines", MODULE_PATH)
pipelines = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipelines)


class _Scheduler:
    order = 1
    timesteps = torch.arange(10)

    def set_begin_index(self, index):
        self.begin_index = index


def test_fill_strength_slices_scheduler_timesteps():
    pipe = SimpleNamespace(scheduler=_Scheduler())

    timesteps, steps = pipelines.FluxFillPipeline.get_timesteps(
        pipe, num_inference_steps=10, strength=0.4, device=torch.device("cpu")
    )

    assert timesteps.tolist() == [6, 7, 8, 9]
    assert steps == 4
    assert pipe.scheduler.begin_index == 6


class _StrengthWasRouted(Exception):
    pass


class _VAE:
    def enable_tiling(self):
        pass


class _RoutingProbe:
    vae = _VAE()
    vae_scale_factor = 8
    default_sample_size = 64
    scheduler = SimpleNamespace(order=1)
    _execution_device = torch.device("cpu")

    def _check_inputs(self, *args, **kwargs):
        pass

    def get_batch_size(self, prompt):
        return 1

    def prepare_timesteps(self, num_inference_steps, height, width, device):
        return torch.arange(num_inference_steps), num_inference_steps

    def get_timesteps(self, num_inference_steps, strength, device):
        self.routed_strength = strength
        raise _StrengthWasRouted


def test_shared_inpainting_path_routes_strength_to_timestep_slicing():
    pipe = _RoutingProbe()

    with pytest.raises(_StrengthWasRouted):
        pipelines.FluxPipeline._call_shared(
            pipe,
            prompt="panorama",
            is_inpainting=True,
            image=object(),
            mask_image=object(),
            strength=0.4,
        )

    assert pipe.routed_strength == 0.4
