# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
import torch
from transformers import AutoModelForImageTextToText

from vllm.platforms import current_platform
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

from ....conftest import HfRunner, ImageTestAssets, VllmRunner
from .vlm_utils import model_utils

MODEL = "google/gemma-3-4b-it"
PROMPT = (
    "<bos><start_of_turn>user\n"
    "<start_of_image>What is the content in the center of the image?"
    "<end_of_turn>\n<start_of_turn>model\n"
)
TINY_MODEL = "hf-internal-testing/tiny-random-Gemma3ForConditionalGeneration"

GIT_MODEL = "microsoft/git-base-coco"
GIT_PROMPT = "a photo of"


def _install_prefill_hidden_capture(model):
    model = getattr(model, "module", model)
    model._prefill_hidden = None

    language_model = model.language_model.model
    original_forward = language_model.forward

    def forward(*args, **kwargs):
        hidden_states = original_forward(*args, **kwargs)
        if model._prefill_hidden is None and torch.is_tensor(hidden_states):
            # Capturing hidden states for comparison is a deliberate D2H.
            with gpu_sync_allowed():
                model._prefill_hidden = hidden_states.detach().float().cpu()
        return hidden_states

    language_model.forward = forward


def _get_prefill_hidden(model):
    model = getattr(model, "module", model)
    hidden = getattr(model, "_prefill_hidden", None)
    assert hidden is not None
    return hidden


def _get_hf_prefill_hidden(hf_model: HfRunner, image: Any):
    inputs = hf_model.get_inputs([PROMPT], images=[image])[0]
    with torch.no_grad():
        outputs = hf_model.model.model(
            **hf_model.wrap_device(inputs),
            use_cache=False,
        )
    return outputs.last_hidden_state[0].detach().float().cpu()


def _get_vllm_prefill_hidden(
    vllm_runner: type[VllmRunner],
    model: str,
    image: Any,
    vllm_runner_kwargs: dict[str, Any],
):
    with vllm_runner(
        model,
        max_model_len=4096,
        max_num_seqs=2,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
        **vllm_runner_kwargs,
    ) as vllm_model:
        vllm_model.apply_model(_install_prefill_hidden_capture)
        vllm_model.generate_greedy([PROMPT], max_tokens=1, images=[image])
        return vllm_model.apply_model(_get_prefill_hidden)[0]


@pytest.mark.core_model
@pytest.mark.skipif(
    current_platform.is_rocm(), reason="ROCm attention has accuracy issue for this test"
)
def test_mm_prefix_lm_e2e(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    image_assets: ImageTestAssets,
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression: Gemma3 native prefill must apply image prefix-LM mask."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    image = image_assets[0].pil_image

    vllm_runner_kwargs: dict[str, Any] = {
        "mm_processor_cache_gb": 0,
        "mm_processor_kwargs": {"do_pan_and_scan": True},
    }
    vllm_hidden = _get_vllm_prefill_hidden(
        vllm_runner, MODEL, image, vllm_runner_kwargs
    )

    hf_model = hf_runner(
        MODEL,
        auto_cls=AutoModelForImageTextToText,
    )
    hf_model = model_utils.gemma3_patch_hf_runner(hf_model)

    with hf_model:
        hf_hidden = _get_hf_prefill_hidden(hf_model, image)

    assert vllm_hidden.shape == hf_hidden.shape

    full_cos = torch.nn.functional.cosine_similarity(
        vllm_hidden.flatten(), hf_hidden.flatten(), dim=0
    )
    image_cos = torch.nn.functional.cosine_similarity(
        vllm_hidden[1:769].flatten(), hf_hidden[1:769].flatten(), dim=0
    )

    assert full_cos > 0.9, (
        "Gemma3 mm-prefix-LM full prefill hidden states should be close to HF; "
        f"got {full_cos=}"
    )
    assert image_cos > 0.9, (
        "Gemma3 mm-prefix-LM image prefill hidden states should be close to HF; "
        f"got {image_cos=}"
    )


@pytest.mark.cpu_model
def test_mm_prefix_lm_e2e_tiny(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    image_assets: ImageTestAssets,
    monkeypatch: pytest.MonkeyPatch,
):
    """Same check in fp32 with a tiny Gemma3, so it runs on CPU."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    image = image_assets[0].pil_image

    vllm_hidden = _get_vllm_prefill_hidden(
        vllm_runner, TINY_MODEL, image, {"dtype": "float32"}
    )
    with hf_runner(
        TINY_MODEL, dtype="float32", auto_cls=AutoModelForImageTextToText
    ) as hf_model:
        hf_hidden = _get_hf_prefill_hidden(hf_model, image)

    torch.testing.assert_close(vllm_hidden, hf_hidden, atol=1e-3, rtol=1e-3)


@pytest.mark.cpu_model
def test_mm_prefix_lm_e2e_git(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    image_assets: ImageTestAssets,
    monkeypatch: pytest.MonkeyPatch,
):
    """GIT's image tokens attend to each other bidirectionally. Its prefill
    hidden states only match HF if the attention backend applies that mask."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    image = image_assets[0].pil_image

    with vllm_runner(
        GIT_MODEL,
        dtype="float32",
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
    ) as vllm_model:
        vllm_model.apply_model(_install_prefill_hidden_capture)
        vllm_model.generate_greedy([GIT_PROMPT], max_tokens=1, images=[image])
        vllm_hidden = vllm_model.apply_model(_get_prefill_hidden)[0]

    hf_model = model_utils.git_patch_hf_runner(hf_runner(GIT_MODEL, dtype="float32"))
    with hf_model:
        inputs = hf_model.get_inputs([GIT_PROMPT], images=[image])[0]
        with torch.no_grad():
            hf_hidden = hf_model.model.git(**hf_model.wrap_device(inputs))
        hf_hidden = hf_hidden.last_hidden_state[0].float().cpu()

    torch.testing.assert_close(vllm_hidden, hf_hidden, atol=1e-3, rtol=1e-3)
