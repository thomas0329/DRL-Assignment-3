import torch.nn as nn
import torch
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=4):
        # state (4, 84, 84)
        super(FeatureExtractor, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # Layer 1
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),              # Layer 2
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),              # Layer 3
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),              # Layer 4
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        x = self.conv_layers(x) # [1, 32, 6, 6]
        return self.flatten(x)
    
class InverseModel(nn.Module):
    def __init__(self, action_sz):
        super(InverseModel, self).__init__()
        self.fc = nn.Sequential(    # [1, 2304]
            nn.Linear(2304, 256),
            nn.Linear(256, action_sz),
        )
        
    def forward(self ,x):
        return self.fc(x)
    

class ForwardModel(nn.Module):
    def __init__(self, state_feat_dim=1152, action_dim=12):
        super(ForwardModel, self).__init__()
        self.input_dim = state_feat_dim + action_dim  # 1152 + 12 = 1164

        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 288)
        self.fc3 = nn.Linear(288, state_feat_dim)  # Predict next φ(sₜ₊₁), also 1152-dim

    def forward(self, phi_st_flat, action):
        action_onehot = F.one_hot(action, num_classes=12).float()  # [1, 12]
        # phi_st_flat: [batch_size, 1152]
        # action_onehot: [batch_size, 12]
        x = torch.cat([phi_st_flat, action_onehot], dim=1)  # [batch_size, 1164]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # [batch_size, 1152]
        return x


class ICM(nn.Module):
    def __init__(self):
        super(ICM, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.feat_extractor = FeatureExtractor().to(device=self.device)
        self.inverse_model = InverseModel(action_sz=12).to(device=self.device)
        self.forward_model = ForwardModel().to(device=self.device)
        self.optimizer = torch.optim.Adam(
            list(self.feat_extractor.parameters()) +
            list(self.inverse_model.parameters()) +
            list(self.forward_model.parameters())
        )
        self.CE = nn.CrossEntropyLoss()
        self.beta = 0.2
        self.eta = 7905000

    def forward(self, state, next_state, action):
        state = self.to_model_input(state)
        next_state = self.to_model_input(next_state)
        
        state_feat = self.feat_extractor(state) # [1, 32*6*6]
        next_state_feat = self.feat_extractor(next_state)
        feat = torch.cat((state_feat, next_state_feat), dim=1)    # [1, 2304]
        action_distr = self.inverse_model(feat) # [1, 12]

        # train feat extractor, inverse model, and forward model
        
        action = torch.tensor([action]).to(device=self.device)

        inverse_loss = self.CE(action_distr, action)
        next_state_feat_pred = self.forward_model(state_feat, action)
        forward_loss = F.mse_loss(next_state_feat_pred, next_state_feat)
        total_loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        intrinsic_rwd = self.eta * forward_loss.item()

        return intrinsic_rwd













        

    def to_model_input(self, state):
        state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        return state