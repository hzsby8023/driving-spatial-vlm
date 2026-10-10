"""Autoregressive continuous local SE(2) trajectory decoder; no action codebook."""
import torch
from torch import nn
import torch.nn.functional as F


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def pose_to_local(traj):
    """Current-frame absolute future poses [B,T,3] -> body-frame increments."""
    prev = torch.cat((torch.zeros_like(traj[:, :1]), traj[:, :-1]), dim=1)
    d = traj[..., :2] - prev[..., :2]
    yaw = prev[..., 2]
    c, s = yaw.cos(), yaw.sin()
    dx = c * d[..., 0] + s * d[..., 1]
    dy = -s * d[..., 0] + c * d[..., 1]
    return torch.stack((dx, dy, wrap(traj[..., 2] - yaw)), dim=-1)


def local_to_pose(local):
    """Differentiable sequential integration of body-frame local actions."""
    pos = torch.zeros_like(local[:, 0, :2])
    yaw = torch.zeros_like(local[:, 0, 2])
    out = []
    for t in range(local.shape[1]):
        dx, dy, da = local[:, t].unbind(-1)
        c, s = yaw.cos(), yaw.sin()
        pos = pos + torch.stack((c * dx - s * dy, s * dx + c * dy), dim=-1)
        yaw = wrap(yaw + da)
        out.append(torch.cat((pos, yaw[:, None]), dim=-1))
    return torch.stack(out, dim=1)


def cv_trajectory(history, history_times, future_times):
    """Current ego-frame CV + constant yaw-rate baseline, supports irregular timestamps."""
    dt = (history_times[:, -1] - history_times[:, -2]).clamp_min(1e-3)
    vel = (history[:, -1, :2] - history[:, -2, :2]) / dt[:, None]
    rate = wrap(history[:, -1, 2] - history[:, -2, 2]) / dt
    return torch.cat((vel[:, None] * future_times[..., None],
                      wrap(rate[:, None] * future_times)[..., None]), dim=-1)


class ContinuousGRU(nn.Module):
    """VLM context + history -> autoregressive residual local action decoder."""
    def __init__(self, vision_dim, hidden=256, steps=10):
        super().__init__()
        self.steps = steps
        self.context = nn.Sequential(nn.Linear(vision_dim + 16 + 2, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.input_proj = nn.Linear(3 + 3 + 1, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.out = nn.Linear(hidden, 3)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, vision, history, history_times, future_times, gt=None, teacher_forcing=0.0):
        """gt only supplies previous action during teacher forcing, never vision/context."""
        batch, steps = future_times.shape
        if steps != self.steps:
            raise ValueError(f'Expected {self.steps} future steps, got {steps}')
        cv_pose = cv_trajectory(history, history_times, future_times)
        cv_local = pose_to_local(cv_pose)
        status_dt = (history_times[:, -1] - history_times[:, -2]).clamp_min(1e-3)
        accel = (history[:, -1, 3] - history[:, -2, 3]) / status_dt
        status = torch.stack((history[:, -1, 3], accel), dim=-1)
        h = self.context(torch.cat((vision.float(), history.reshape(batch, -1), status), dim=-1))
        prev_action = torch.zeros(batch, 3, device=vision.device, dtype=torch.float32)
        gt_local = pose_to_local(gt) if gt is not None else None
        actions = []
        for t in range(steps):
            delta_t = future_times[:, t] - (future_times[:, t-1] if t else 0.)
            x = torch.cat((prev_action, cv_local[:, t], delta_t[:, None]), dim=-1)
            h = self.gru(F.silu(self.input_proj(x)), h)
            # Residual around CV: stable zero-initialized initial predictions.
            action = cv_local[:, t] + self.out(h)
            actions.append(action)
            if gt_local is not None and teacher_forcing > 0:
                mask = (torch.rand(batch, 1, device=vision.device) < teacher_forcing)
                prev_action = torch.where(mask, gt_local[:, t], action)
            else:
                prev_action = action
        actions = torch.stack(actions, dim=1)
        return local_to_pose(actions), actions, cv_pose


def trajectory_loss(pred, gt, local, local_gt):
    pos = F.smooth_l1_loss(pred[..., :2], gt[..., :2])
    yaw = (1 - torch.cos(pred[..., 2] - gt[..., 2])).mean()
    action = F.smooth_l1_loss(local[..., :2], local_gt[..., :2])
    return pos + 0.3 * action + 0.2 * yaw
