
import torch


def planning_metrics(pred, target):
    """pred, target: [B, T, 3] in current ego frame."""
    distance = torch.linalg.vector_norm(
        pred[..., :2] - target[..., :2],
        dim=-1,
    )

    return {
        "ade": distance.mean().item(),
        "fde": distance[:, -1].mean().item(),
    }


def constant_velocity_baseline(
    ego_history,
    history_times,
    future_times,
):
    """
    ego_history: [B, 4, 4] = x, y, yaw, speed
    history_times: [B, 4], relative to current time
    future_times: [B, 6], relative to current time
    """
    dt = (
        history_times[:, -1]
        - history_times[:, -2]
    ).clamp_min(1e-4)

    # Positions are already expressed in current ego frame.
    xy_velocity = (
        ego_history[:, -1, :2]
        - ego_history[:, -2, :2]
    ) / dt[:, None]

    yaw_rate = torch.atan2(
        torch.sin(
            ego_history[:, -1, 2]
            - ego_history[:, -2, 2]
        ),
        torch.cos(
            ego_history[:, -1, 2]
            - ego_history[:, -2, 2]
        ),
    ) / dt

    future_xy = (
        xy_velocity[:, None, :]
        * future_times[..., None]
    )

    future_yaw = (
        yaw_rate[:, None]
        * future_times
    )

    future_yaw = torch.atan2(
        torch.sin(future_yaw),
        torch.cos(future_yaw),
    )

    return torch.cat(
        [future_xy, future_yaw.unsqueeze(-1)],
        dim=-1,
    )
