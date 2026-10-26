"""Record episodes as GIFs, optionally overlaying the ego-attention weights.

Intermediate simulation frames are captured through highway-env's video hook
(``set_record_video_wrapper``), so the GIF is smooth even at 1 decision per second.
With an ego-attention policy, a line is drawn from the ego to every observed vehicle,
its width and opacity proportional to the attention weight.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


class _FrameGrabber:
    """Duck-typed stand-in for gymnasium's RecordVideo, as expected by highway-env."""

    def __init__(self, base_env, decorate):
        self.base_env = base_env
        self.decorate = decorate
        self.frames: list[Image.Image] = []
        self.frames_per_sec = 15

    def _capture_frame(self):
        frame = self.base_env.render()
        if frame is not None:
            self.frames.append(self.decorate(frame))


def attention_extractor(model):
    """Return the EgoAttentionExtractor of an SB3 model, or None."""
    from safelane.agents.extractors import EgoAttentionExtractor

    policy = getattr(model, "policy", None)
    for name in ("features_extractor", "pi_features_extractor"):
        ext = getattr(policy, name, None)
        if isinstance(ext, EgoAttentionExtractor):
            return ext
    return None


def observed_vehicles(env) -> list:
    """Vehicles in the same order as the Kinematics observation rows (row 0 = ego)."""
    u = env.unwrapped
    obs_type = u.observation_type
    others = u.road.close_objects_to(
        obs_type.observer_vehicle,
        u.PERCEPTION_DISTANCE,
        count=obs_type.vehicles_count - 1,
        see_behind=obs_type.see_behind,
        sort=obs_type.order == "sorted",
        vehicles_only=not obs_type.include_obstacles,
    )
    return [obs_type.observer_vehicle, *others[-obs_type.vehicles_count + 1 :]]


def record_gif(
    policy,
    env,
    path: str,
    seed: int = 0,
    model=None,
    max_steps: int = 100,
    caption: str | None = None,
    fps: int = 15,
) -> dict:
    """Run one episode with ``policy(obs, env)`` and save a GIF. Returns the episode metrics."""
    u = env.unwrapped
    ext = attention_extractor(model) if model is not None else None
    state = {"weights": None, "vehicles": [], "text": ""}

    def decorate(frame: np.ndarray) -> Image.Image:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img, "RGBA")
        surface = u.viewer.sim_surface if u.viewer is not None else None
        if state["weights"] is not None and surface is not None:
            ego = state["vehicles"][0]
            ex, ey = surface.pos2pix(*ego.position)
            w = state["weights"]
            w = w / max(w.max(), 1e-6)
            for weight, v in zip(w[1:], state["vehicles"][1:]):
                if weight < 0.05:
                    continue
                vx, vy = surface.pos2pix(*v.position)
                draw.line([(ex, ey), (vx, vy)], fill=(220, 40, 40, int(60 + 195 * weight)),
                          width=max(1, int(1 + 5 * weight)))
        if state["text"]:
            draw.rectangle([0, 0, img.width, 14], fill=(0, 0, 0, 140))
            draw.text((4, 1), state["text"], fill=(255, 255, 255, 255))
        return img

    grabber = _FrameGrabber(u, decorate)
    obs, _ = env.reset(seed=seed)
    u.render()
    u.set_record_video_wrapper(grabber)
    action_names = {v: k for k, v in u.action_type.actions_indexes.items()}
    info = {}
    for t in range(max_steps):
        action = policy(obs, env)
        if ext is not None and ext.last_attention is not None:
            state["weights"] = ext.last_attention[0].mean(dim=0).cpu().numpy()
            state["vehicles"] = observed_vehicles(env)
        label = caption + " | " if caption else ""
        state["text"] = f"{label}t={t}s  v={u.vehicle.speed:4.1f} m/s  action={action_names.get(action, action)}"
        obs, _, terminated, truncated, info = env.step(action)
        grabber._capture_frame()
        if terminated or truncated:
            break
    frames = grabber.frames
    if frames:
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0,
                       optimize=True)
    u._record_video_wrapper = None
    return info.get("episode_metrics", {})
