import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Encoder(nn.Module):
    """
    Encoder: CNN + MLP
    Input: o_t ∈ R^(3×64×64)
    Output: e_t ∈ R^(d_e) (e.g., 128)
    """
    def __init__(self, embedding_dim=128):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.flatten_dim = 256 * 4 * 4
        
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim),
        )
    
    def forward(self, obs):
        """
        Args:
            obs: (B, 3, 64, 64) tensor
        Returns:
            e_t: (B, embedding_dim) tensor
        """
        x = self.cnn(obs)
        e_t = self.mlp(x)
        return e_t


class RSSM(nn.Module):
    """
    RSSM latent dynamics
    one determinstic state and one stochastic state
    """
    def __init__(self, action_dim, embedding_dim=128, hidden_dim=200, stochastic_dim=64):
        super(RSSM, self).__init__()
        self.hidden_dim = hidden_dim
        self.stochastic_dim = stochastic_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        
        # prior network
        self.prior_mlp = nn.Sequential(
            nn.Linear(stochastic_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        self.prior_mean = nn.Linear(hidden_dim, stochastic_dim)
        self.prior_std = nn.Linear(hidden_dim, stochastic_dim)
        
        # correction network
        self.posterior_mlp = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.posterior_mean = nn.Linear(hidden_dim, stochastic_dim)
        self.posterior_std = nn.Linear(hidden_dim, stochastic_dim)
    
    def prior(self, h_prev, z_prev, a_prev):
        """
        Prior (prediction): p(z_t | h_{t-1}, z_{t-1}, a_{t-1})
        
        Args:
            h_prev: (B, hidden_dim) previous deterministic state
            z_prev: (B, stochastic_dim) previous stochastic state
            a_prev: (B, action_dim) previous action
        Returns:
            h_t: (B, hidden_dim) new deterministic state
            z_t_prior: (B, stochastic_dim) sampled prior stochastic state
            prior_dist: Normal distribution for KL loss
        """
        # concat previous stochastic state and action
        x = torch.cat([z_prev, a_prev], dim=-1)
        
        # prior network
        x = self.prior_mlp(x)
        h_t = self.gru(x, h_prev)
        
        mean = self.prior_mean(h_t)
        log_std = self.prior_std(h_t)
        log_std = torch.clamp(log_std, min=-10, max=2) 
        std = torch.exp(log_std)
        
        prior_dist = Normal(mean, std)
        z_t_prior = prior_dist.rsample() 
        
        return h_t, z_t_prior, prior_dist
    
    def posterior(self, h_t, e_t):
        """
        Posterior (correction): q(z_t | h_t, e_t)
        
        Args:
            h_t: (B, hidden_dim) deterministic state
            e_t: (B, embedding_dim) encoded observation
        Returns:
            z_t_post: (B, stochastic_dim) sampled posterior stochastic state
            post_dist: Normal distribution for KL loss
        """
        # concat deterministic state and encoded observation
        x = torch.cat([h_t, e_t], dim=-1)
        
        # correction network
        x = self.posterior_mlp(x)
        mean = self.posterior_mean(x)
        log_std = self.posterior_std(x)
        log_std = torch.clamp(log_std, min=-10, max=2) 
        std = torch.exp(log_std)
        
        post_dist = Normal(mean, std)
        z_t_post = post_dist.rsample() 
        
        return z_t_post, post_dist


class Decoder(nn.Module):
    """
    Image recon
    Output: o_hat_t 
    """
    def __init__(self, hidden_dim=200, stochastic_dim=64):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.stochastic_dim = stochastic_dim
        
        # mlp
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + stochastic_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256 * 4 * 4),  
            nn.ReLU(),
        )
        
        # deconv layers
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  
        )
    
    def forward(self, h_t, z_t):
        """
        Args:
            h_t: (B, hidden_dim) deterministic state
            z_t: (B, stochastic_dim) stochastic state
        Returns:
            o_hat_t: (B, 3, 64, 64) reconstructed observation
        """
        # concat deterministic state and stochastic state
        x = torch.cat([h_t, z_t], dim=-1)
        
        # mlp
        x = self.mlp(x)
        x = x.view(-1, 256, 4, 4)
        
        # deconv layers
        o_hat_t = self.deconv(x)
        
        return o_hat_t


class RewardHead(nn.Module):
    """
    Reward pred
    """
    def __init__(self, hidden_dim=200, stochastic_dim=64):
        super(RewardHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + stochastic_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
    
    def forward(self, h_t, z_t):
        """
        Args:
            h_t: (B, hidden_dim) deterministic state
            z_t: (B, stochastic_dim) stochastic state
        Returns:
            r_hat_t: (B, 1) predicted reward
        """
        x = torch.cat([h_t, z_t], dim=-1)
        r_hat_t = self.mlp(x)
        return r_hat_t.squeeze(-1)  


class ValueHead(nn.Module):
    """
    Value head
    """
    def __init__(self, hidden_dim=200, stochastic_dim=64):
        super(ValueHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + stochastic_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
    
    def forward(self, h_t, z_t):
        """
        """
        x = torch.cat([h_t, z_t], dim=-1)
        v_hat_t = self.mlp(x)
        return v_hat_t.squeeze(-1)  # (B,)


class PolicyPriorHead(nn.Module):
    """
    Policy prior used for MCTS with PUCT
    """
    def __init__(self, hidden_dim=200, stochastic_dim=64, action_dim=3):
        super(PolicyPriorHead, self).__init__()
        self.action_dim = action_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + stochastic_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
    
    def forward(self, h_t, z_t):
        """
        """
        x = torch.cat([h_t, z_t], dim=-1)
        logits = self.mlp(x)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


class WorldModel(nn.Module):
    """
    worldmodel
    - Encoder
    - RSSM (prior & posterior)
    - Decoder
    - Reward head
    - Value head
    - Policy prior head
    """
    def __init__(
        self,
        action_dim=3,
        embedding_dim=128,
        hidden_dim=200,
        stochastic_dim=64,
        action_space_size=3,
    ):
        super(WorldModel, self).__init__()
        
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.stochastic_dim = stochastic_dim
        
        self.encoder = Encoder(embedding_dim=embedding_dim)
        self.rssm = RSSM(
            action_dim=action_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            stochastic_dim=stochastic_dim,
        )
        self.decoder = Decoder(
            hidden_dim=hidden_dim,
            stochastic_dim=stochastic_dim,
        )
        self.reward_head = RewardHead(
            hidden_dim=hidden_dim,
            stochastic_dim=stochastic_dim,
        )
        self.value_head = ValueHead(
            hidden_dim=hidden_dim,
            stochastic_dim=stochastic_dim,
        )
        self.policy_prior_head = PolicyPriorHead(
            hidden_dim=hidden_dim,
            stochastic_dim=stochastic_dim,
            action_dim=action_space_size,
        )
    
    def forward(self, obs, action, h_prev=None, z_prev=None, use_posterior=True):
        """
        Forward pass through the beautiful world

        Args:
            obs: (B, 3, 64, 64) current observation
            action: (B, action_dim) previous action (one-hot or embedding)
            h_prev: (B, hidden_dim) previous deterministic state (None for first step)
            z_prev: (B, stochastic_dim) previous stochastic state (None for first step)
            use_posterior: bool, if True use posterior (training), else use prior (imagination)
        
        Returns:
            dict with all outputs
        """
        batch_size = obs.shape[0]
        device = obs.device
        
        # initialize states if None
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_dim, device=device)
        if z_prev is None:
            z_prev = torch.zeros(batch_size, self.stochastic_dim, device=device)
        
        # encode observation
        e_t = self.encoder(obs)
        
        # prior prediction
        h_t, z_t_prior, prior_dist = self.rssm.prior(h_prev, z_prev, action)
        
        # posterior correction (if training)
        if use_posterior:
            z_t, post_dist = self.rssm.posterior(h_t, e_t)
        else:
            z_t = z_t_prior
            post_dist = None
        
        # image reconstruction
        o_hat_t = self.decoder(h_t, z_t)
        
        # reward prediction
        r_hat_t = self.reward_head(h_t, z_t)
        
        # value prediction
        v_hat_t = self.value_head(h_t, z_t)
        
        # policy prior
        policy_logits, policy_probs = self.policy_prior_head(h_t, z_t)
        
        return {
            'e_t': e_t,
            'h_t': h_t,
            'z_t': z_t,
            'z_t_prior': z_t_prior,
            'prior_dist': prior_dist,
            'post_dist': post_dist,
            'o_hat_t': o_hat_t,
            'r_hat_t': r_hat_t,
            'v_hat_t': v_hat_t,
            'policy_logits': policy_logits,
            'policy_probs': policy_probs,
        }
    
    def forward_sequence(self, obs, action, h_prev=None, z_prev=None):
        """
        Efficient forward pass for a sequence of observations.
        Batches the heavy Encoder and Decoder operations.
        
        Args:
            obs: (B, T, 3, 64, 64)
            action: (B, T, action_dim) - action[t] is action taken at time t (input for t+1)
            h_prev: (B, hidden_dim) initial state
            z_prev: (B, stochastic_dim) initial state
        
        Returns:
            dict with stacked outputs
        """
        B, T = obs.shape[:2]
        device = obs.device
        
        # 1. Batch Encode
        # Flatten time dimension: (B*T, C, H, W)
        obs_flat = obs.view(B * T, *obs.shape[2:])
        e_seq_flat = self.encoder(obs_flat)
        e_seq = e_seq_flat.view(B, T, -1)
        
        # 2. RSSM Loop (Sequential)
        if h_prev is None:
            h_prev = torch.zeros(B, self.hidden_dim, device=device)
        if z_prev is None:
            z_prev = torch.zeros(B, self.stochastic_dim, device=device)
            
        h_seq = []
        z_seq = []
        prior_means = []
        prior_stds = []
        post_means = []
        post_stds = []
        
        # Current states
        h_t = h_prev
        z_t = z_prev
        
        for t in range(T):
            # For RSSM step t, we need action_{t-1}. 
            # Assuming action sequence is aligned such that action[:, t] is the action taken at step t.
            # The dynamics are: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
            
            if t == 0:
                # For the first step, we usually assume a zero action or provided prev action
                # Here we assume zero action for simplicity of the sequence
                a_prev = torch.zeros(B, self.action_dim, device=device)
            else:
                a_prev = action[:, t-1]
            
            # Prior: Predict t from t-1
            h_t, z_t_prior, prior_dist = self.rssm.prior(h_t, z_t, a_prev)
            
            # Posterior: Correct using encoded observation at t
            e_t = e_seq[:, t]
            z_t, post_dist = self.rssm.posterior(h_t, e_t)
            
            # Store
            h_seq.append(h_t)
            z_seq.append(z_t)
            
            prior_means.append(prior_dist.mean)
            prior_stds.append(prior_dist.stddev)
            post_means.append(post_dist.mean)
            post_stds.append(post_dist.stddev)
            
        # Stack temporal dimension
        h_seq = torch.stack(h_seq, dim=1) # (B, T, H)
        z_seq = torch.stack(z_seq, dim=1) # (B, T, Z)
        
        # Create batched distributions
        prior_dist_seq = Normal(torch.stack(prior_means, dim=1), torch.stack(prior_stds, dim=1))
        post_dist_seq = Normal(torch.stack(post_means, dim=1), torch.stack(post_stds, dim=1))
        
        # 3. Batch Decode & Heads
        h_flat = h_seq.view(B * T, -1)
        z_flat = z_seq.view(B * T, -1)
        
        o_hat_flat = self.decoder(h_flat, z_flat)
        r_hat_flat = self.reward_head(h_flat, z_flat)
        v_hat_flat = self.value_head(h_flat, z_flat)
        policy_logits_flat, policy_probs_flat = self.policy_prior_head(h_flat, z_flat)
        
        return {
            'o_hat_t': o_hat_flat.view(B, T, *obs.shape[2:]),
            'r_hat_t': r_hat_flat.view(B, T),
            'v_hat_t': v_hat_flat.view(B, T),
            'policy_logits': policy_logits_flat.view(B, T, -1),
            'policy_probs': policy_probs_flat.view(B, T, -1),
            'prior_dist': prior_dist_seq,
            'post_dist': post_dist_seq,
            'h_t': h_t, # final state
            'z_t': z_t, # final state
        }
    
    def compute_loss(self, obs, action, reward, value_targets=None, h_prev=None, z_prev=None, 
                     recon_loss_weight=1.0, reward_loss_weight=1.0, kl_loss_weight=0.1, value_loss_weight=1.0, free_nats=0.0):
        """
        Compute training losses:
        - Reconstruction loss (MSE)
        - Reward prediction loss (MSE)
        - KL loss (rssm prior (used for imagination) vs rssm posterior (used for training))
        - Value prediction loss (MSE)
        
        Args:
            obs: (B, 3, 64, 64) observation
            action: (B, action_dim) action
            reward: (B,) true reward
            value_targets: (B,) n-step return targets (optional)
            h_prev, z_prev: previous states
            recon_loss_weight: weight for reconstruction loss
            reward_loss_weight: weight for reward loss
            kl_loss_weight: weight for KL loss
            value_loss_weight: weight for value loss
        
        Returns:
            dict with losses
        """
        # forward pass with posterior (training)
        outputs = self.forward(obs, action, h_prev, z_prev, use_posterior=True)
        
        # reconstruction loss (MSE)
        recon_loss = F.mse_loss(outputs['o_hat_t'], obs)
        
        # reward loss
        reward_loss = F.mse_loss(outputs['r_hat_t'], reward)
        
        # KL loss 
        kl_loss = 0.0
        if outputs['prior_dist'] is not None and outputs['post_dist'] is not None:
            kl_loss = torch.distributions.kl.kl_divergence(
                outputs['post_dist'], outputs['prior_dist']
            ).mean()
        
        # value loss
        value_loss = torch.tensor(0.0, device=obs.device)
        if value_targets is not None:
            value_loss = F.mse_loss(outputs['v_hat_t'], value_targets)
        
        # total loss
        total_loss = (
            recon_loss_weight * recon_loss +
            reward_loss_weight * reward_loss +
            kl_loss_weight * kl_loss +
            value_loss_weight * value_loss
        )
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'reward_loss': reward_loss,
            'kl_loss': kl_loss,
            'value_loss': value_loss,
        }
