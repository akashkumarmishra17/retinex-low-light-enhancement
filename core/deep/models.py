"""
Deep Retinex Neural Network Architectures (PyTorch).

Implements:
1. DecomNet: Illumination-Reflectance decomposition network with weight sharing.
2. RelightNet: Deep multi-scale illumination adjustment network.
3. RetinexLoss: Comprehensive loss suite combining reconstruction, mutual consistency,
                and structure-aware illumination smoothness.

References:
    Wei, C., Wang, W., Yang, W., & Liu, J. (2018).
    "Deep retinex decomposition for low-light enhancement."
    BMVC 2018.

    Zhang, Y., Zhang, J., & Guo, X. (2019).
    "Kindling the darkness: A practical low-light image enhancer."
    ACM Multimedia 2019.

Author: Akash Kumar Mishra (Roll No: 24CS003388)
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecomNet(nn.Module):
    """
    Decomposition Network (DecomNet).
    
    Decomposes an observed image S(x, y) into its physical components:
    - Reflectance R(x, y) in [0, 1]^3
    - Illumination I(x, y) in [0, 1]^1
    
    Architecture:
        5-layer Convolutional network with ReLU activations and Sigmoid output gating.
    """
    def __init__(self, in_channels: int = 3, num_filters: int = 64):
        super(DecomNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(num_filters, 4, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W) in [0, 1].
            
        Returns:
            Tuple of:
                - R (torch.Tensor): Reflectance of shape (B, 3, H, W)
                - I (torch.Tensor): Illumination of shape (B, 1, H, W)
        """
        feat = self.relu(self.conv1(x))
        feat = self.relu(self.conv2(feat))
        feat = self.relu(self.conv3(feat))
        feat = self.relu(self.conv4(feat))
        out = self.sigmoid(self.conv_out(feat))
        
        # Split into 3-channel Reflectance R and 1-channel Illumination I
        R = out[:, 0:3, :, :]
        I = out[:, 3:4, :, :]
        
        return R, I


class RelightNet(nn.Module):
    """
    Relighting Network (RelightNet).
    
    Takes the restored reflectance R and low-light illumination I_low to synthesize 
    an enhanced, uniformly distributed illumination field I_enhanced.
    
    Architecture:
        U-Net style encoder-decoder with multi-scale skip connections to aggregate
        both global contextual illumination and local edge details.
    """
    def __init__(self, in_channels: int = 4, num_filters: int = 64):
        super(RelightNet, self).__init__()
        
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.enc2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, stride=2, padding=1)
        
        # Bottleneck
        self.mid = nn.Conv2d(num_filters * 2, num_filters * 2, kernel_size=3, stride=1, padding=1)
        
        # Decoder with Skip Connections
        self.up1 = nn.ConvTranspose2d(num_filters * 2, num_filters, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.Conv2d(num_filters * 2, num_filters, kernel_size=3, stride=1, padding=1)
        
        self.up2 = nn.ConvTranspose2d(num_filters, num_filters, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.Conv2d(num_filters * 2, num_filters, kernel_size=3, stride=1, padding=1)
        
        # Final illumination projection
        self.conv_out = nn.Conv2d(num_filters, 1, kernel_size=3, stride=1, padding=1)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, R: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            R (torch.Tensor): Reflectance tensor of shape (B, 3, H, W).
            I (torch.Tensor): Low-light illumination tensor of shape (B, 1, H, W).
            
        Returns:
            torch.Tensor: Enhanced illumination field I_enhanced of shape (B, 1, H, W).
        """
        x = torch.cat([R, I], dim=1)  # Shape (B, 4, H, W)
        
        e1 = self.lrelu(self.enc1(x))
        e2 = self.lrelu(self.enc2(e1))
        e3 = self.lrelu(self.enc3(e2))
        
        m = self.lrelu(self.mid(e3))
        
        u1 = self.lrelu(self.up1(m))
        d1 = self.lrelu(self.dec1(torch.cat([u1, e2], dim=1)))
        
        u2 = self.lrelu(self.up2(d1))
        d2 = self.lrelu(self.dec2(torch.cat([u2, e1], dim=1)))
        
        # Output illumination field bounded strictly in [0, 1]
        I_relit = self.sigmoid(self.conv_out(d2))
        return I_relit


class RetinexLoss(nn.Module):
    """
    Composite Loss Function for Retinex-based Deep Decomposition.
    
    Loss Formulation:
        L_total = L_recon + lambda_mutual * L_mutual + lambda_smooth * L_smooth
        
    Where:
        - L_recon: Reconstruction fidelity ||S - R ⊙ I||_1
        - L_mutual: Invariant reflectance consistency ||R_low - R_norm||_1
        - L_smooth: Structure-aware illumination smoothness:
                    ||grad(I) ⊙ exp(-alpha * grad(S))||_1
    """
    def __init__(self, lambda_mutual: float = 0.09, lambda_smooth: float = 0.15, alpha: float = 10.0):
        super(RetinexLoss, self).__init__()
        self.lambda_mutual = lambda_mutual
        self.lambda_smooth = lambda_smooth
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()

    def gradient_x(self, img: torch.Tensor) -> torch.Tensor:
        return torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])

    def gradient_y(self, img: torch.Tensor) -> torch.Tensor:
        return torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])

    def structure_aware_smoothness(self, I: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """
        Penalize gradients in illumination map except where sharp edges occur in S.
        """
        # Convert S to grayscale luminance for edge guide
        S_gray = 0.299 * S[:, 0:1, :, :] + 0.587 * S[:, 1:2, :, :] + 0.114 * S[:, 2:3, :, :]
        
        grad_I_x = self.gradient_x(I)
        grad_I_y = self.gradient_y(I)
        
        grad_S_x = self.gradient_x(S_gray)
        grad_S_y = self.gradient_y(S_gray)
        
        weight_x = torch.exp(-self.alpha * grad_S_x)
        weight_y = torch.exp(-self.alpha * grad_S_y)
        
        loss_x = torch.mean(grad_I_x * weight_x)
        loss_y = torch.mean(grad_I_y * weight_y)
        
        return loss_x + loss_y

    def forward(
        self, 
        S_low: torch.Tensor, 
        R_low: torch.Tensor, 
        I_low: torch.Tensor,
        S_norm: torch.Tensor = None, 
        R_norm: torch.Tensor = None, 
        I_norm: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        
        # 1. Low-light reconstruction loss: S_low ≈ R_low ⊙ I_low
        I_low_3ch = I_low.repeat(1, 3, 1, 1)
        loss_recon_low = self.l1_loss(S_low, R_low * I_low_3ch)
        
        loss_total = loss_recon_low
        loss_mutual = torch.tensor(0.0, device=S_low.device)
        loss_recon_norm = torch.tensor(0.0, device=S_low.device)
        
        # 2. Normal-light reconstruction and mutual reflectance constraint (when paired)
        if S_norm is not None and R_norm is not None and I_norm is not None:
            I_norm_3ch = I_norm.repeat(1, 3, 1, 1)
            loss_recon_norm = self.l1_loss(S_norm, R_norm * I_norm_3ch)
            loss_mutual = self.l1_loss(R_low, R_norm)
            loss_total = loss_total + loss_recon_norm + self.lambda_mutual * loss_mutual
            
        # 3. Structure-aware illumination smoothness
        loss_smooth = self.structure_aware_smoothness(I_low, S_low)
        loss_total = loss_total + self.lambda_smooth * loss_smooth
        
        return {
            "loss_total": loss_total,
            "loss_recon_low": loss_recon_low,
            "loss_recon_norm": loss_recon_norm,
            "loss_mutual": loss_mutual,
            "loss_smooth": loss_smooth
        }
