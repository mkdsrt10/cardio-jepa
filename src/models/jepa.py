import copy
import torch
import torch.nn as nn

def apply_jepa_masking(x: torch.Tensor, mask_type="both", mask_ratio_time=0.25, mask_leads_count=3) -> torch.Tensor:
    """Applies Context-Target masking along the temporal and/or lead dimension.
    
    x shape: [B, 12, Length]
    Returns: x_masked of same shape as x.
    """
    x_masked = x.clone()
    B, Leads, Length = x.shape
    
    # 1. Lead-wise masking
    if mask_type in ["lead", "both"]:
        for b in range(B):
            # Select random leads to mask out completely
            mask_leads = torch.randperm(Leads)[:mask_leads_count]
            x_masked[b, mask_leads, :] = 0.0
            
    # 2. Time block masking (continuous temporal segment zeroed across all leads)
    if mask_type in ["time", "both"]:
        mask_len = int(Length * mask_ratio_time)
        for b in range(B):
            # Select random starting position for temporal block masking
            start_idx = torch.randint(0, Length - mask_len + 1, (1,)).item()
            x_masked[b, :, start_idx : start_idx + mask_len] = 0.0
            
    return x_masked

def update_ema_variables(model: nn.Module, ema_model: nn.Module, alpha: float):
    """Exponential Moving Average (EMA) parameter update helper.
    
    theta_target = alpha * theta_target + (1 - alpha) * theta_context
    """
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1.0 - alpha)

def sigreg_loss(z: torch.Tensor, lambd: float = 20.0) -> torch.Tensor:
    """Signature Regularization (SIGReg) covariance and variance loss.
    
    Guarantees high-dimensional representations, preventing dimensional collapse.
    z shape: [B, D]
    """
    B, D = z.shape
    if B <= 1:
        return torch.tensor(0.0, device=z.device)
        
    # Center representations around mean
    z_mean = z - z.mean(dim=0, keepdim=True)
    
    # Compute empirical covariance matrix of shape [D, D]
    cov = (z_mean.T @ z_mean) / (B - 1)
    
    # 1. Variance Loss: Push each diagonal element's std deviation to be >= 1.0
    diag = torch.diagonal(cov)
    std_diag = torch.sqrt(diag + 1e-4)
    var_loss = torch.mean(torch.clamp(1.0 - std_diag, min=0.0))
    
    # 2. Covariance Loss: Push off-diagonal elements to 0.0 to decorrelate dimensions
    off_diag = cov - torch.diag(diag)
    cov_loss = (off_diag ** 2).sum() / D
    
    # Regularization loss scaled by lambda
    return lambd * (var_loss + cov_loss)

class JEPAPredictor(nn.Module):
    """Predictor network that maps context representation to target representation space."""
    
    def __init__(self, latent_dim=256, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, z_context):
        # Input shape: [B, latent_dim]
        # Output shape: [B, latent_dim]
        return self.net(z_context)

class ECGJEPA(nn.Module):
    """ECG-JEPA Self-Supervised Learning Framework."""
    
    def __init__(self, context_encoder, latent_dim=256, predictor_hidden_dim=512, ema_decay=0.996):
        super().__init__()
        self.context_encoder = context_encoder
        self.ema_decay = ema_decay
        
        # Clone context encoder to build target encoder (no gradients)
        self.target_encoder = copy.deepcopy(context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        # Predictor mapping context representation -> target representation
        self.predictor = JEPAPredictor(latent_dim=latent_dim, hidden_dim=predictor_hidden_dim)
        
    def forward_context(self, x_masked):
        """Processes partially masked waveform through context encoder & predictor."""
        z_context = self.context_encoder(x_masked)
        z_predicted = self.predictor(z_context)
        return z_predicted, z_context
        
    @torch.no_grad()
    def forward_target(self, x):
        """Processes full, clean waveform through target encoder (EMA weights)."""
        z_target = self.target_encoder(x)
        return z_target
        
    def update_target_ema(self):
        """Updates target encoder weights using exponential moving average."""
        update_ema_variables(self.context_encoder, self.target_encoder, self.ema_decay)
