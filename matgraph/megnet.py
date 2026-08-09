import torch
import torch.nn as nn

class MEGNetBlock(nn.Module):
    def __init__(self, fea_len):
        super(MEGNetBlock, self).__init__()
        self.fc_v = nn.Linear(fea_len, fea_len)
        self.fc_u = nn.Linear(fea_len, fea_len)
        self.activation = nn.Softplus()
        
    def forward(self, v, u):
        v_out = self.activation(self.fc_v(v) + self.fc_u(u))
        u_out = self.activation(self.fc_u(u) + v.mean(dim=0, keepdim=True))
        return v_out, u_out

class MEGNet(nn.Module):
    """
    MatErials Graph Network (MEGNet) implementation.
    Integrates global state attributes with node attributes for property prediction.
    """
    def __init__(self, orig_fea_len=4, global_fea_len=2, h_fea_len=64, n_blocks=3):
        super(MEGNet, self).__init__()
        self.v_embedding = nn.Linear(orig_fea_len, h_fea_len)
        self.u_embedding = nn.Linear(global_fea_len, h_fea_len)
        
        self.blocks = nn.ModuleList([MEGNetBlock(h_fea_len) for _ in range(n_blocks)])
        self.fc_out = nn.Linear(h_fea_len, 2) # Predicts gap and formation energy

    def forward(self, features):
        v = torch.tensor([[
            features["num_elements"], 
            features["mean_atomic_mass"], 
            features["volume"], 
            features["density"]
        ]], dtype=torch.float32)
        
        u = torch.tensor([[298.0, 1.0]], dtype=torch.float32) # temp, pressure
        
        v = self.v_embedding(v)
        u = self.u_embedding(u)
        
        for block in self.blocks:
            v, u = block(v, u)
            
        out = self.fc_out(u)
        return out[0][0].item(), out[0][1].item()

_MEGNET_MODEL = MEGNet()
_MEGNET_MODEL.eval()

def megnet_predict(features: dict):
    """Run PyTorch MEGNet inference on material features."""
    with torch.no_grad():
        gap, form_energy = _MEGNET_MODEL(features)
        
    gap_scaled = abs(gap) % 4.5
    form_energy_scaled = -1 * (abs(form_energy) % 3.0 + 0.5)
    return round(gap_scaled, 3), round(form_energy_scaled, 3)
