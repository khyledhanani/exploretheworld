import torch
import torch.nn.functional as F
import numpy as np


class MinMaxStats:
    """Stores the min-max values of the tree for normalizing Q-values."""
    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class MCTSNode:
    def __init__(self, h, z, prior_prob=None, parent=None, parent_action=None):
        """
        One search-tree node corresponding to a latent state (h, z).
        prior_prob: 1D tensor/array-like of length action_dim with prior over actions.
        """
        self.h = h  # (hidden_dim,)
        self.z = z  # (stochastic_dim,)

        # Store prior on the same device as h/z to avoid CPU<->device sync.
        if isinstance(prior_prob, torch.Tensor):
            self.prior = prior_prob.detach().to(device=self.h.device, dtype=torch.float32)
        else:
            self.prior = torch.tensor(prior_prob, device=self.h.device, dtype=torch.float32)
        # Normalize defensively (can be slightly off due to numerical noise)
        s = self.prior.sum()
        if torch.isfinite(s) and float(s.item()) > 0:
            self.prior = self.prior / s

        self.parent = parent
        self.parent_action = parent_action
        self.reward_parent = 0.0  # reward obtained from parent via parent_action

        action_dim = int(self.prior.shape[0])
        self.children = [None for _ in range(action_dim)]
        # Store stats as tensors for cheaper math + easier aggregation
        self.N = torch.zeros(action_dim, device=self.h.device, dtype=torch.int32)       # visit counts
        self.W = torch.zeros(action_dim, device=self.h.device, dtype=torch.float32)    # total value
        self.Q = torch.zeros(action_dim, device=self.h.device, dtype=torch.float32)    # mean value

        self.is_expanded = False

    def select_action(self, c_puct, min_max_stats):
        # NOTE: we keep this as a small Python loop since action_dim is tiny (3).
        best_score, best_a = float("-inf"), None

        total_N = float(self.N.sum().item())
        sqrt_total_N = np.sqrt(total_N) if total_N > 0 else 0.0

        for a_idx in range(int(self.prior.shape[0])):
            n = int(self.N[a_idx].item())
            if n > 0:
                q = float(self.Q[a_idx].item())
                q_val = float(min_max_stats.normalize(q))
            else:
                q_val = 0.0

            prior_a = float(self.prior[a_idx].item())
            u_val = float(c_puct) * prior_a * sqrt_total_N / (1.0 + n)
            score = q_val + u_val

            if score > best_score:
                best_score = score
                best_a = a_idx

        return best_a


def expand_and_evaluate(world_model, node, action):
    """
    Expand the tree at (node, action) using the world model and
    return the scalar value estimate of the resulting leaf state.
    """
    device = node.h.device
    action_dim = int(node.prior.shape[0])

    # one-hot action on the correct device
    a_tensor = torch.tensor([action], device=device, dtype=torch.long)
    a_one_hot = F.one_hot(a_tensor, num_classes=action_dim).float()  # (1, A)

    # RSSM prior: latent dynamics step
    h_next, z_next, _ = world_model.rssm.prior(
        node.h.unsqueeze(0),  # (1, H)
        node.z.unsqueeze(0),  # (1, Z)
        a_one_hot,            # (1, A)
    )
    h_next = h_next.squeeze(0)
    z_next = z_next.squeeze(0)

    # Predict reward, value and policy prior from the new latent state
    r_hat = world_model.reward_head(h_next.unsqueeze(0), z_next.unsqueeze(0))      # (1,)
    v_hat = world_model.value_head(h_next.unsqueeze(0), z_next.unsqueeze(0))      # (1,)
    _, policy_probs = world_model.policy_prior_head(
        h_next.unsqueeze(0), z_next.unsqueeze(0)
    )  # (1, A)

    policy_probs = policy_probs.squeeze(0).detach()

    # Create and attach child node
    child = MCTSNode(
        h=h_next.detach(),
        z=z_next.detach(),
        prior_prob=policy_probs,
        parent=node,
        parent_action=action,
    )
    child.reward_parent = float(r_hat.item())
    child.is_expanded = True
    node.children[action] = child

    # Return leaf value for backup (scalar)
    return float(v_hat.item())


def backup(path, leaf_value, discount, min_max_stats):
    """
    Backup value from the leaf to the root along the given path.
    path: list of (node, action) from root to leaf-parent.
    """
    G = leaf_value
    for node, action in reversed(path):
        child = node.children[action]
        r = child.reward_parent
        G = r + discount * G

        node.N[action] += 1
        node.W[action] += float(G)
        node.Q[action] = node.W[action] / node.N[action].to(dtype=torch.float32)
        
        # Update MinMax stats for correct normalization in future selections
        min_max_stats.update(float(node.Q[action].item()))


def MCTS(world_model, root_h, root_z, c_puct, num_simulations, discount, action_space_size, temperature=1.0, dirichlet_alpha=0.3):
    """
    Run MuZero-style MCTS starting from latent state (root_h, root_z).
    Expects root_h, root_z to be 1D tensors on the same device as world_model.
    """
    min_max_stats = MinMaxStats()
    
    with torch.no_grad():
        # Get root prior and value
        _, policy_probs = world_model.policy_prior_head(
            root_h.unsqueeze(0), root_z.unsqueeze(0)
        )
        v_root = world_model.value_head(root_h.unsqueeze(0), root_z.unsqueeze(0))
        min_max_stats.update(float(v_root.item()))

        policy_probs = policy_probs.squeeze(0).detach()
        
        # Add Dirichlet noise to root priors for exploration.
        # NOTE: torch's Dirichlet sampling is not implemented on MPS in some PyTorch builds,
        # so we sample with NumPy and move to the model device.
        noise_np = np.random.dirichlet([float(dirichlet_alpha)] * int(action_space_size)).astype(np.float32)
        noise = torch.tensor(noise_np, device=root_h.device, dtype=torch.float32)
        policy_probs = 0.75 * policy_probs + 0.25 * noise
        s = policy_probs.sum()
        if torch.isfinite(s) and float(s.item()) > 0:
            policy_probs = policy_probs / s

        root = MCTSNode(root_h.detach(), root_z.detach(), policy_probs, None, None)
        root.is_expanded = True

        for _ in range(num_simulations):
            node = root
            path = []
            
            # Selection
            while node.is_expanded:
                a = node.select_action(c_puct, min_max_stats)
                path.append((node, a))
                if node.children[a] is None:
                    break
                node = node.children[a]

            # Expansion
            last_node, last_action = path[-1]
            leaf_value = expand_and_evaluate(world_model, last_node, last_action)

            # Backup
            backup(path, leaf_value, discount, min_max_stats)

    visit_counts = root.N.to(dtype=torch.float32)

    if float(visit_counts.sum().item()) > 0:
        if float(temperature) == 0.0:
            policy_target = torch.zeros_like(visit_counts)
            policy_target[int(torch.argmax(visit_counts).item())] = 1.0
        else:
            # Temperature scaling
            visit_counts_temp = visit_counts.pow(1.0 / float(temperature))
            policy_target = visit_counts_temp / visit_counts_temp.sum()
    else:
        policy_target = torch.ones_like(visit_counts) / float(visit_counts.shape[0])

    # Search-improved root value target: pi · Q(root, ·)
    root_value = float((policy_target * root.Q).sum().item())

    # Choose action based on the policy target
    action = int(torch.multinomial(policy_target, num_samples=1).item())

    return action, policy_target.detach().cpu().numpy(), root_value