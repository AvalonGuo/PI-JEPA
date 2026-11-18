import torch
import torch.nn as nn
import torchvision
import copy
import os

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint

# post-catgen log hygiene (\u5019\u9009B): DR3D_QUIET=1 \u65f6\u540c\u4e00 banner \u6bcf\u8fdb\u7a0b\u53ea\u6253\u5370\u4e00\u6b21
# (eval_policy.sh \u4f1a setdefault('1'); \u8bad\u7ec3\u94fe\u8def\u672a\u8bbe\u8be5\u53d8\u91cf -> \u884c\u4e3a\u4e0d\u53d8)
_BANNER_SEEN = set()


def _banner_once(key):
    if os.environ.get("DR3D_QUIET") == "1":
        if key in _BANNER_SEEN:
            return False
        _BANNER_SEEN.add(key)
    return True


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules




class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 pool_type: str='max',
                 pool_temperature: float=1.0,
                 pool_score_hidden: int=0,
                 state_dim: int=0,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
            pool_type (str): global point aggregation: 'max' (PointNet baseline),
                'mean', or 'softmax' (learned attention pooling).
            pool_temperature (float): softmax only, score /= temperature (<1 = sharper).
            pool_score_hidden (int): softmax only, 0 = Linear(C->C); >0 = 2-layer MLP.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        if _banner_once("pointnet_layernorm"):
            cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        if _banner_once("pointnet_final_norm"):
            cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        if pool_type not in ('max', 'mean', 'softmax'):
            raise NotImplementedError(f"pool_type: {pool_type} (use max|mean|softmax)")
        self.pool_type = pool_type
        self.pool_temperature = float(pool_temperature)
        self.pc_dim = block_channel[-1]
        self.state_dim = int(state_dim)   # kept for call-site compatibility, unused
        self.score_hidden = int(pool_score_hidden)

        # softmax (attention) pooling: one score per (point, channel), softmax over N.
        self.score_mlp = None
        if pool_type == 'softmax':
            c = block_channel[-1]
            in_dim = c
            if self.score_hidden > 0:
                self.score_mlp = nn.Sequential(
                    nn.Linear(in_dim, self.score_hidden),
                    nn.ReLU(),
                    nn.Linear(self.score_hidden, c),
                )
                last = self.score_mlp[-1]
            else:
                self.score_mlp = nn.Linear(in_dim, c)
                last = self.score_mlp
            # zero-init -> uniform softmax weights at step 0 (== mean-pool); the
            # layer only has to learn how far to sharpen the point attention.
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        if _banner_once("pointnet_pool"):
            cprint("[PointNetEncoderXYZ] pool_type: {} (temperature={}, score_hidden={})"
                   .format(pool_type, self.pool_temperature, self.score_hidden), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            if _banner_once("pointnet_no_projection"):
                cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def _score_logits(self, feats, state=None):
        """Per-(point, channel) logits -> (B, N, C); softmax only.

        ``state`` is accepted for call-site compatibility and is not used.
        """
        return self.score_mlp(feats)

    def pool_scores(self, feats, state=None):
        """Pooling logits (B, N, C) (softmax only)."""
        return self._score_logits(feats) / max(self.pool_temperature, 1e-6)

    def pool_weights(self, feats, state=None):
        """Pooling weights normalized over points; (B, N, C)."""
        return torch.softmax(self.pool_scores(feats), dim=1)

    def _pool(self, x, state=None):
        """(B, N, C) point features -> (B, C) global feature."""
        if self.pool_type == 'mean':
            return torch.mean(x, 1)
        if self.pool_type == 'softmax':
            return (self.pool_weights(x) * x).sum(dim=1)
        return torch.max(x, 1)[0]

    def forward(self, x, state=None):
        x = self.mlp(x)
        x = self._pool(x, state)
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()

    


class DP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        
        
        if _banner_once("dp3_pc_shape"):
            cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        if _banner_once("dp3_state_shape"):
            cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        if _banner_once("dp3_imagination"):
            cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        state_out_dim = int(state_mlp_size[-1])
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg,
                                                    state_dim=state_out_dim)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        if _banner_once("dp3_output_dim"):
            cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)

        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64

        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        # state_feat is concat'ed to the encoder output (not used for pooling)
        pn_feat = self.extractor(points, state_feat)    # B * out_channel

        final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels