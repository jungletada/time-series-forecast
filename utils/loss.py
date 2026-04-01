import torch
import torch.nn as nn
import torch.fft

class FirstOrderDiffLoss(nn.Module):
    """一阶差分损失"""
    def __init__(self):
        super(FirstOrderDiffLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        """
        x: Predicted sequence [Batch, Length, Channel]
        y: Ground truth sequence [Batch, Length, Channel]
        """
        # Calculate first-order differences along the time dimension (dim=1)
        diff_x = x[:, 1:, :] - x[:, :-1, :]
        diff_y = y[:, 1:, :] - y[:, :-1, :]
        
        # Compute MSE between the differences
        return self.mse(diff_x, diff_y)
    

class FrequencyDomainLoss(nn.Module):
    def __init__(self, k_dominant=5):
        """
        k_dominant: Number of top frequency components to consider.
        """
        super(FrequencyDomainLoss, self).__init__()
        self.k = k_dominant
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        """
        x: Predicted sequence [Batch, Length, Channel]
        y: Ground truth sequence [Batch, Length, Channel]
        """
        # Apply FFT to convert to frequency domain
        # rfft returns complex tensor representing positive frequencies
        fft_x = torch.fft.rfft(x, dim=1)
        fft_y = torch.fft.rfft(y, dim=1)
        
        # Calculate amplitude spectrum
        amp_x = torch.abs(fft_x)
        amp_y = torch.abs(fft_y)
        
        # Identify top-k dominant frequencies from Ground Truth (y)
        # We sort by amplitude to find indices of top-k
        # dims: [Batch, Freq_Len, Channel] -> sort along Freq_Len (dim=1)
        sorted_indices = torch.argsort(amp_y, dim=1, descending=True)
        
        # Indices for top-k (Dominant)
        top_k_indices = sorted_indices[:, :self.k, :]
        
        # Indices for the rest (Noise)
        noise_indices = sorted_indices[:, self.k:, :]
        
        # Gather frequencies based on indices
        # We use gather to select specific frequency components from the amplitude spectra
        dom_amp_x = torch.gather(amp_x, 1, top_k_indices)
        dom_amp_y = torch.gather(amp_y, 1, top_k_indices)
        
        noise_amp_x = torch.gather(amp_x, 1, noise_indices)
        # Ground truth noise components are implicitly targets, but the paper says 
        # "minimizing non-dominant frequency magnitudes", suggesting we want prediction's noise to be low (or close to GT's noise).
        # Standard interpretation: MSE on dominant parts + MSE on noise parts
        noise_amp_y = torch.gather(amp_y, 1, noise_indices)

        # L_dom: MSE of dominant frequencies
        l_dom = self.mse(dom_amp_x, dom_amp_y)
        
        # L_noise: MSE of noise frequencies
        l_noise = self.mse(noise_amp_x, noise_amp_y)
        
        # Total Frequency Loss, scaled by 1/sqrt(T) as per paper 
        T = x.shape[1]
        return (1.0 / (T ** 0.5)) * (l_dom + l_noise)
    

class TemporalFeatureExtractor(nn.Module):
    """
    Placeholder for the pre-trained feature extractor described in the paper.
    It typically consists of a Transformer block and an MLP.
    """
    def __init__(self, input_dim, d_model=64, d_z=32):
        super(TemporalFeatureExtractor, self).__init__()
        # Simple implementation based on paper description [cite: 895]
        self.transformer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.projector = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_z) # Maps to latent dim d_z
        )
        self.input_proj = nn.Linear(input_dim, d_model) # Project input to d_model

    def forward(self, x):
        # x: [Batch, Length, Channel] -> Project to d_model
        x_proj = self.input_proj(x) 
        
        # Pass through Transformer
        x_trans = self.transformer(x_proj)
        
        # Map to latent feature vector z
        # Usually we take the mean or last token, paper says "maps output into feature vector"
        # Assuming pooling over time dimension for a single vector per sequence
        x_pool = torch.mean(x_trans, dim=1) 
        
        z = self.projector(x_pool)
        return z


class PerceptualFeatureLoss(nn.Module):
    def __init__(self, feature_extractor):
        super(PerceptualFeatureLoss, self).__init__()
        self.f_time = feature_extractor
        
        # Freeze the feature extractor parameters [cite: 833]
        for param in self.f_time.parameters():
            param.requires_grad = False
        
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        """
        x, y: [Batch, Length, Channel]
        """
        # Extract features z_x and z_y [cite: 914]
        z_x = self.f_time(x)
        z_y = self.f_time(y)
        
        # Compute MSE between features, normalized by latent dimension d_z [cite: 915]
        d_z = z_x.shape[-1]
        loss = self.mse(z_x, z_y) # MSELoss already averages, so we might not need explicit 1/d_z if using 'mean' reduction
        
        # If strict adherence to formula 1/d_z * ||z_x - z_y||^2 is needed (which is sum of squared errors / d_z):
        # MSELoss(reduction='mean') is exactly 1/N * sum((x-y)^2). 
        # Since z is a vector of size d_z, MSELoss computes exactly the paper's formula.
        return loss
    
    
class SATLLoss(nn.Module):
    def __init__(self, 
                 feature_extractor, 
                 alpha=0.2, 
                 beta=0.2, 
                 gamma=0.1, 
                 delta=0.5,
                 k_dominant=5):
        """
        Hyperparameters defaults based on paper[cite: 976]:
        alpha=0.2, beta=0.2, gamma=0.1, delta=0.5
        """
        super(SATLLoss, self).__init__()
        
        # Hyperparameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        # Sub-losses
        self.l_diff = FirstOrderDiffLoss()
        self.l_freq = FrequencyDomainLoss(k_dominant=k_dominant)
        # self.l_perceptual = PerceptualFeatureLoss(feature_extractor)
        self.l_mse = nn.MSELoss()

    def forward(self, x, y):
        """
        x: Prediction
        y: Ground Truth
        """
        # 1. Calculate individual geometric losses
        loss_diff = self.l_diff(x, y)
        loss_freq = self.l_freq(x, y)
        # loss_perceptual = self.l_perceptual(x, y)
        
        # 2. Calculate SATL [cite: 917]
        # L_SATL = alpha * L_diff + beta * L_freq + gamma * L_perceptual
        l_satl = (self.alpha * loss_diff) + \
                 (self.beta * loss_freq)
                 # (self.gamma * loss_perceptual)
        
        # 3. Calculate MSE Loss
        loss_mse = self.l_mse(x, y)
        
        # 4. Calculate Total Loss [cite: 918]
        # L_total = L_SATL + delta * L_MSE
        l_total = l_satl + (self.delta * loss_mse)
        
        return l_total
    

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # 这里的 1e-12 是为了防止数值溢出
        loss = torch.log(torch.cosh(y_pred - y_true + 1e-12))
        return torch.mean(loss)
    
    
class WeightedL1Loss:
    def __init__(self, alpha, loss_mode):
        self.alpha = alpha
        self.loss_mode = loss_mode
        if self.loss_mode == 'L1':
            self.loss_fun = nn.L1Loss(reduction='none')
        elif self.loss_mode == 'L2':
            self.loss_fun = nn.MSELoss(reduction='none')
        elif self.loss_mode == 'L1L2':
            self.loss_fun1 = nn.L1Loss(reduction='none')
            self.loss_fun2 = nn.MSELoss(reduction='none')

    def __call__(self, pred, gt):
        # [b,l,n]
        if pred.ndim == 1:
            # imputation
            mask = torch.isnan(gt)
            if torch.any(mask):
                # pred, gt = pred.masked_fill(mask, 0), gt.masked_fill(mask, 0)
                pred, gt = pred[~mask], gt[~mask]

            loss_fun = nn.L1Loss(reduction='mean')
            weightedLoss = loss_fun(pred, gt)
        else:
            L = pred.shape[1]
            weights = (torch.tensor([(i + 1) ** (-self.alpha) for i in range(L)]).unsqueeze(dim=0).unsqueeze(dim=-1)
                       .to(pred.device))
            if self.loss_mode in ['L1', 'L2']:
                loss_vec = self.loss_fun(pred, gt)
                weightedLoss = torch.mean(loss_vec * weights)
            elif self.loss_mode == 'L1L2':
                loss_vec = self.loss_fun1(pred, gt)
                loss_vec2 = self.loss_fun2(pred, gt)
                weightedLoss = torch.mean(loss_vec * weights + loss_vec2 * weights)
            else:
                raise NotImplementedError
        return weightedLoss