"""Albumentations augmentation pipeline for archaeological hole detection.

Provides light and heavy augmentation profiles configurable via Hydra config,
with bbox-safe pipelines that integrate seamlessly with Ultralytics YOLO training.

Integration strategies
---------------------
**A. ``model.train(augmentations=...)`` kwarg (recommended)**

  Ultralytics already has built-in support for Albumentations via its
  ``v8_transforms`` function, which includes an ``Albumentations`` step that
  reads ``hyp.augmentations``.  Passing custom transforms as the
  ``augmentations`` kwarg to ``model.train()`` is the simplest integration::

      from ultralytics import YOLO
      from scripts.augmentation import get_pipeline

      pipeline = get_pipeline("heavy")
      model = YOLO("models/yolo26n.pt")
      model.train(data="dataset/data.yaml", epochs=100,
                  augmentations=pipeline.transforms)

  The :func:`get_ultralytics_augmentation` helper builds the complete kwargs
  dict from a Hydra config, so in ``train.py``::

      aug_kwargs = get_ultralytics_augmentation(cfg)
      model.train(data="...", **aug_kwargs)

  works automatically because ``v8_transforms`` reads ``hyp.augmentations``.

**B. YOLODataset monkey-patch (for when you need guaranteed insertion)**

  Call :func:`patch_yolodataset` before model creation to replace
  ``YOLODataset.build_transforms`` with a version that injects Albumentations
  transforms into ``hyp.augmentations``, which are then picked up by the
  built-in ``Albumentations`` step in ``v8_transforms``::

      from scripts.augmentation import patch_yolodataset
      patch_yolodataset(mode="heavy")

      from ultralytics import YOLO
      model = YOLO("models/yolo26n.pt")
      model.train(...)  # ← automatically uses Albumentations

  You can also set the ``AUGMENTATION_MODE`` environment variable to
  ``"light"`` or ``"heavy"`` *before* importing ``scripts.augmentation``
  to trigger auto-patching at import time (see :func:`_auto_patch`).

Usage
-----
    >>> from scripts.augmentation import get_pipeline
    >>> p = get_pipeline("heavy")
    >>> result = p(image=img, bboxes=[[0.5, 0.5, 0.2, 0.2]], class_labels=[0])
    >>> result["image"].shape
    (640, 640, 3)
    >>> result["bboxes"]
    [[0.5, 0.5, 0.2, 0.2]]

    >>> from omegaconf import OmegaConf
    >>> cfg = OmegaConf.create({
    ...     "augmentation": {
    ...         "enabled": True,
    ...         "mode": "heavy",
    ...         "ultralytics": {"hsv_h": 0.015, "hsv_s": 0.7}
    ...     }
    ... })
    >>> get_ultralytics_augmentation(cfg)  # doctest: +SKIP
    {'hsv_h': 0.015, 'hsv_s': 0.7, 'augmentations': [...]}
"""

from __future__ import annotations

import warnings

# Suppress albumentations >= 2.0 deprecation warnings for transforms we
# deliberately keep (ShiftScaleRotate) and API migration notes.
warnings.filterwarnings("ignore", message=".*ShiftScaleRotate.*Affine.*")
warnings.filterwarnings("ignore", message=".*Argument.*not valid.*")

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Albumentations import guard
# ---------------------------------------------------------------------------

try:
    import albumentations as A
    from albumentations.core.composition import Compose

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:  # pragma: no cover
    A = None  # type: ignore[assignment]
    Compose = None  # type: ignore[assignment,misc]
    ALBUMENTATIONS_AVAILABLE = False
    logger.warning(
        "Albumentations is not installed.  Install it with:\n"
        "  pip install albumentations>=1.0.3\n"
        "Augmentation pipelines will be unavailable."
    )

# ---------------------------------------------------------------------------
# Pipeline builders — exact transform specs from configs/augmentation/{light,heavy}.yaml
# ---------------------------------------------------------------------------


def _light_transforms() -> list[A.BasicTransform]:
    """Return the list of light Albumentations transforms.

    Matches ``configs/augmentation/light.yaml`` exactly.
    All transforms use explicit ``p=`` probability
    (never ``always_apply=True``).
    """
    return [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
        A.HueSaturationValue(p=0.2, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
        A.Blur(p=0.2, blur_limit=3),
    ]


def _heavy_transforms() -> list[A.BasicTransform]:
    """Return the list of heavy Albumentations transforms.

    Includes all light transforms plus spatial and noise-based augmentations.
    Matches ``configs/augmentation/heavy.yaml`` exactly.
    """
    return [
        # ── Light transforms (included) ──────────────────────────────
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
        A.HueSaturationValue(p=0.2, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
        A.Blur(p=0.2, blur_limit=3),
        # ── Heavy-specific transforms ────────────────────────────────
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            p=0.5,
            shift_limit_x=0.1,
            shift_limit_y=0.1,
            scale_limit=0.2,
            rotate_limit=45,
            border_mode=0,  # cv2.BORDER_CONSTANT
        ),
        A.RandomGamma(p=0.3, gamma_limit=(80, 120)),
        A.CLAHE(p=0.2, clip_limit=4.0, tile_grid_size=(8, 8)),
        # NOTE: API changed in albumentations >= 2.0:
        #   max_holes → num_holes_range, max_height/max_width → hole_height/width_range, fill_value → fill
        A.CoarseDropout(
            p=0.3,
            num_holes_range=(1, 8),
            hole_height_range=(64, 64),
            hole_width_range=(64, 64),
            fill=0,
        ),
        # NOTE: intensity & color_shift are ranges (tuples) in albumentations >= 2.0
        A.ISONoise(p=0.2, intensity=(0.1, 0.3), color_shift=(0.03, 0.07)),
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_pipeline(mode: str = "light") -> Optional[Compose]:
    """Return an Albumentations ``Compose`` pipeline for the given *mode*.

    Parameters
    ----------
    mode : str
        One of ``"light"``, ``"heavy"``, or ``"none"``.

        * ``"light"`` — HorizontalFlip, RandomBrightnessContrast,
          HueSaturationValue, Blur.
        * ``"heavy"`` — All light transforms plus RandomRotate90,
          ShiftScaleRotate, RandomGamma, CLAHE, CoarseDropout, ISONoise.
        * ``"none"`` — Returns ``None`` (no external augmentation).

    Returns
    -------
    A.Compose or None
        Bbox-safe pipeline configured with ``bbox_params`` set to
        ``A.BboxParams(format='yolo', label_fields=['class_labels'])``.
        Returns ``None`` when *mode* is ``"none"`` or when Albumentations
        is not installed.

    Raises
    ------
    ValueError
        If *mode* is not one of the recognised values.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        logger.error("Albumentations not installed — cannot build pipeline.")
        return None

    if mode == "none":
        return None

    if mode == "light":
        transforms = _light_transforms()
    elif mode == "heavy":
        transforms = _heavy_transforms()
    else:
        raise ValueError(
            f"Unknown augmentation mode: {mode!r}. "
            f"Expected one of: 'light', 'heavy', 'none'."
        )

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def get_ultralytics_augmentation(cfg) -> dict[str, Any]:
    """Build Ultralytics ``model.train()`` kwargs from a Hydra config.

    Extracts **both**:

    1.  Ultralytics native augmentation hyper-parameters from
        ``cfg.augmentation.ultralytics`` (e.g. ``hsv_h``, ``scale``,
        ``fliplr``, ``mosaic``).
    2.  An Albumentations transform list for the ``augmentations`` kwarg
        based on ``cfg.augmentation.mode``.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        The resolved Hydra config.  Must contain an ``augmentation``
        key.  Typically the result of Hydra's ``@hydra.main`` composition.

    Returns
    -------
    dict[str, Any]
        Kwargs suitable for ``YOLO.model.train(**kwargs)``.
        Includes an ``augmentations`` key (``list`` of Albumentations
        transforms or ``None``).

    Examples
    --------
    >>> from omegaconf import OmegaConf
    >>> cfg = OmegaConf.create({
    ...     "augmentation": {
    ...         "enabled": True,
    ...         "mode": "light",
    ...         "ultralytics": {"hsv_h": 0.015}
    ...     }
    ... })
    >>> kwargs = get_ultralytics_augmentation(cfg)
    >>> kwargs["hsv_h"]
    0.015
    >>> kwargs["augmentations"] is not None
    True
    """
    from omegaconf import OmegaConf

    kwargs: dict[str, Any] = {}

    aug = cfg.get("augmentation", None)
    if aug is None or not aug.get("enabled", False):
        return kwargs

    # 1. Ultralytics native augmentation params (hsv, fliplr, mosaic, etc.)
    ultralytics_aug = getattr(aug, "ultralytics", None)
    if ultralytics_aug is not None:
        for k, v in OmegaConf.to_container(ultralytics_aug, resolve=True).items():
            kwargs[k] = v

    # 2. Albumentations transforms list
    mode = aug.get("mode", "none")
    pipeline = get_pipeline(mode)
    kwargs["augmentations"] = pipeline.transforms if pipeline is not None else None

    return kwargs


# ---------------------------------------------------------------------------
# YOLODataset integration (monkey-patch)
# ---------------------------------------------------------------------------

_AUGMENTATION_MODE: str = "none"
"""Module-level augmentation mode.  Set via :func:`patch_yolodataset`."""


def set_augmentation_mode(mode: str) -> None:
    """Set the global augmentation mode for the monkey-patched YOLODataset.

    Parameters
    ----------
    mode : str
        One of ``"light"``, ``"heavy"``, or ``"none"``.
    """
    global _AUGMENTATION_MODE
    _AUGMENTATION_MODE = mode
    logger.debug("Augmentation mode set to: %s", mode)


def patch_yolodataset(mode: str = "light") -> None:
    """Monkey-patch ``YOLODataset.build_transforms`` to inject Albumentations.

    Replaces ``ultralytics.data.dataset.YOLODataset.build_transforms`` with a
    wrapper that injects the Albumentations pipeline into ``hyp.augmentations``
    **before** calling the original method.  This leverages Ultralytics' own
    ``Albumentations`` step in ``v8_transforms``, which reads
    ``hyp.augmentations``.

    Call this **once** at the top of your training script, before creating
    any ``YOLO`` model::

        from scripts.augmentation import patch_yolodataset
        patch_yolodataset(mode="heavy")  # ← applies to all subsequent training

        from ultralytics import YOLO
        model = YOLO("models/yolo26n.pt")
        model.train(...)

    You can also set the ``AUGMENTATION_MODE`` environment variable to
    ``"light"`` or ``"heavy"`` and the patch will be applied automatically
    when ``scripts.augmentation`` is imported (see :func:`_auto_patch`).

    Parameters
    ----------
    mode : str
        Augmentation mode forwarded to :func:`get_pipeline`.
    """
    import ultralytics.data.dataset as _ds_module

    set_augmentation_mode(mode)

    original_build_transforms = _ds_module.YOLODataset.build_transforms

    def _patched_build_transforms(self, hyp=None):
        # Inject our Albumentations transforms into hyp so that Ultralytics'
        # built-in Albumentations step in v8_transforms picks them up.
        if (
            self.augment
            and _AUGMENTATION_MODE not in ("none", "")
            and hyp is not None
        ):
            pipeline = get_pipeline(_AUGMENTATION_MODE)
            if pipeline is not None:
                try:
                    hyp.augmentations = pipeline.transforms
                except AttributeError:
                    # hyp might be a read-only namespace in some edge cases
                    logger.warning(
                        "Could not set hyp.augmentations — "
                        "hyp object does not support attribute assignment."
                    )

        return original_build_transforms(self, hyp)

    _ds_module.YOLODataset.build_transforms = _patched_build_transforms

    # Also patch the re-exported reference if it exists
    try:
        import ultralytics.data as _data_module

        if hasattr(_data_module, "YOLODataset"):
            _data_module.YOLODataset.build_transforms = _patched_build_transforms
    except ImportError:
        pass

    logger.info(
        "Patched YOLODataset.build_transforms (mode: %s) — "
        "Albumentations will be applied during training.",
        mode,
    )


def _auto_patch() -> None:
    """Auto-apply the monkey-patch from the ``AUGMENTATION_MODE`` env var.

    Called once at import time.  Set::

        AUGMENTATION_MODE=light   # or heavy

    before launching your training script, and the monkey-patch will be
    active without any code changes.
    """
    mode = os.environ.get("AUGMENTATION_MODE", "").strip().lower()
    if mode in ("light", "heavy"):
        try:
            patch_yolodataset(mode=mode)
        except Exception as exc:  # pragma: no cover
            logger.warning("Auto-patch failed: %s", exc)


# ---------------------------------------------------------------------------
# Auto-patch on import (if env var is set)
# ---------------------------------------------------------------------------

_auto_patch()
