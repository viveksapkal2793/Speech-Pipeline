"""FGSM adversarial attack for audio classification models."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..utils.audio import estimate_snr_db


@dataclass(slots=True)
class FGSMResult:
    """Adversarial waveform and diagnostics."""

    adversarial: torch.Tensor
    epsilon: float
    snr_db: float


def _snr_bound(audio: torch.Tensor, snr_db: float) -> float:
    signal_power = torch.mean(audio.detach() ** 2).item() + 1e-12
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return float(math.sqrt(noise_power))


def fgsm_attack(
    forward_fn,
    audio: torch.Tensor,
    target: torch.Tensor,
    epsilon: float | None = None,
    snr_db_floor: float = 40.0,
    targeted: bool = False,
) -> FGSMResult:
    """Generate an FGSM perturbation while respecting an SNR floor."""

    if audio.dim() != 1:
        raise ValueError('Expected a single waveform tensor with shape [samples].')
    audio_orig = audio.detach().clone()
    audio_var = audio_orig.clone().detach().requires_grad_(True)
    logits = forward_fn(audio_var.unsqueeze(0))
    if logits.dim() > 2:
        logits = logits.mean(dim=1)
    loss = torch.nn.functional.cross_entropy(logits, target.view(-1))
    if targeted:
        loss = -loss
    loss.backward()
    max_epsilon = _snr_bound(audio_orig, snr_db_floor)
    epsilon_value = float(min(epsilon if epsilon is not None else max_epsilon, max_epsilon))
    perturbation = epsilon_value * audio_var.grad.sign()
    if targeted:
        perturbation = -perturbation
    adversarial = torch.clamp(audio_orig + perturbation, -1.0, 1.0).detach()
    snr = estimate_snr_db(audio_orig.cpu().numpy(), (adversarial - audio_orig).cpu().numpy())
    return FGSMResult(adversarial=adversarial, epsilon=epsilon_value, snr_db=snr)
