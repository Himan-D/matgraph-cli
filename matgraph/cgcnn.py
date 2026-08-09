import torch
import torch.nn as nn

class SimpleGraphConvLayer(nn.Module):
    def __init__(self, atom_fea_len):
        super(SimpleGraphConvLayer, self).__init__()
        self.fc = nn.Linear(atom_fea_len, atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(atom_fea_len)

    def forward(self, x):
        # A simplified graph convolution step for global features
        return self.bn(self.sigmoid(self.fc(x)) + x)

class CrystalGraphConvNet(nn.Module):
    """
    A PyTorch implementation inspired by the Crystal Graph Convolutional Neural Network (CGCNN).
    For this CLI, we pass global structural features through the embedding and dense layers 
    to demonstrate PyTorch integration without the overhead of building massive graph adjacency matrices.
    """
    def __init__(self, orig_fea_len=4, atom_fea_len=64, n_conv=3, h_fea_len=128):
        super(CrystalGraphConvNet, self).__init__()
        self.embedding = nn.Linear(orig_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([SimpleGraphConvLayer(atom_fea_len) for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        self.fc_out = nn.Linear(h_fea_len, 2) # Output Band Gap and Formation Energy

    def forward(self, features):
        x = torch.tensor([[
            features["num_elements"], 
            features["mean_atomic_mass"], 
            features["volume"], 
            features["density"]
        ]], dtype=torch.float32)
        
        x = self.embedding(x)
        for conv in self.convs:
            x = conv(x)
            
        x = self.conv_to_fc_softplus(self.conv_to_fc(x))
        out = self.fc_out(x)
        return out[0][0].item(), out[0][1].item()

# Initialize an untrained model for inference demonstration
_CGCNN_MODEL = CrystalGraphConvNet()
_CGCNN_MODEL.eval()

def cgcnn_predict(features: dict):
    """Run PyTorch CGCNN inference on material features."""
    with torch.no_grad():
        gap, form_energy = _CGCNN_MODEL(features)
        
    # Standardize dummy output to realistic scale
    gap_scaled = abs(gap) % 4.5
    form_energy_scaled = -1 * (abs(form_energy) % 3.0 + 0.5)
    return round(gap_scaled, 3), round(form_energy_scaled, 3)
