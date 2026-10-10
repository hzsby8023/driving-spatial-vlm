import torch
from torch import nn
import torch.nn.functional as F
from vla.evaluation.planning_metrics import (
    constant_velocity_baseline,
)


class DrivingVLA(nn.Module):
    def __init__(
        self,
        backbone,
        hidden_size,
        horizon=6,
        ego_dim=4,
        fusion_dim=512,
    ):
        super().__init__()

        self.backbone = backbone
        self.horizon = horizon

        self.ego_encoder = nn.Sequential(
            nn.Linear(4 * ego_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )

        self.visual_proj = nn.Sequential(
            nn.Linear(hidden_size, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.SiLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim + 128, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.SiLU(),
        )

        self.trajectory_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.SiLU(),
            nn.Linear(256, horizon * 3),
        )

        # Residual head starts from zero.
        nn.init.zeros_(self.trajectory_head[-1].weight)
        nn.init.zeros_(self.trajectory_head[-1].bias)

    def forward(
        self,
        vlm_inputs,
        ego_history,
        history_times,
        future_times,
        train_backbone=True,
    ):
        if train_backbone:
            outputs = self.backbone(
                **vlm_inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        else:
            with torch.no_grad():
                outputs = self.backbone(
                    **vlm_inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )

        hidden = outputs.hidden_states[-1]
        # [B, L, C]

        attention_mask = vlm_inputs["attention_mask"]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)

        pooled = (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1)
        )
        # [B, C]

        visual_feature = self.visual_proj(
            pooled.float()
        )
        # [B, 512]

        ego_feature = self.ego_encoder(
            ego_history.float().flatten(1)
        )
        # [B, 128]

        fused = self.fusion(
            torch.cat(
                [visual_feature, ego_feature],
                dim=-1,
            )
        )
        # [B, 512]

        residual = self.trajectory_head(fused)
        residual = residual.reshape(-1, self.horizon, 3)

        cv = constant_velocity_baseline(
            ego_history,
            history_times,
            future_times,
        )

        xy = cv[..., :2] + residual[..., :2]
        yaw = cv[..., 2] + residual[..., 2]
        yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
        trajectory = torch.cat(
            [xy, yaw.unsqueeze(-1)],
            dim=-1,
        )

        return trajectory


def trajectory_loss(pred, target):
    position_loss = F.smooth_l1_loss(
        pred[..., :2],
        target[..., :2],
    )

    yaw_delta = pred[..., 2] - target[..., 2]

    yaw_loss = (
        1.0 - torch.cos(yaw_delta)
    ).mean()

    total = position_loss + 0.5 * yaw_loss

    return total, {
        "position": position_loss.detach().item(),
        "yaw": yaw_loss.detach().item(),
    }