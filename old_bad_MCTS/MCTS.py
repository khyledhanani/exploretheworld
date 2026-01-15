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
        prior_prob: 1D array-like of length action_dim with prior over actions.
        """
        self.h = h  # (hidden_dim,)
        self.z = z  # (stochastic_dim,)

        # store prior as a numpy array for convenience
        self.prior = np.array(prior_prob, dtype=np.float32)

        self.parent = parent
        self.parent_action = parent_action
        self.reward_parent = 0.0  # reward obtained from parent via parent_action

        action_dim = len(self.prior)
        self.children = [None for _ in range(action_dim)]
        self.N = [0 for _ in range(action_dim)]     # visit counts
        self.W = [0.0 for _ in range(action_dim)]   # total value
        self.Q = [0.0 for _ in range(action_dim)]   # mean value

        self.is_expanded = False

    def select_action(self, c_puct, min_max_stats):
        best_score, best_a = float("-inf"), None
        
        for a_idx in range(len(self.prior)):
            # Normalize Q value for PUCT
            if self.N[a_idx] > 0:
                q_val = min_max_stats.normalize(self.Q[a_idx])
            else:
                q_val = 0.0 # Unvisited nodes are often treated as having neutral/loss value or max value depending on strategy
            
            total_N = sum(self.N)
            u_val = c_puct * self.prior[a_idx] * np.sqrt(total_N) / (1 + self.N[a_idx])
            
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
    action_dim = len(node.prior)

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

    policy_probs = policy_probs.squeeze(0).detach().cpu().numpy()

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
        node.W[action] += G
        node.Q[action] = node.W[action] / node.N[action]
        
        # Update MinMax stats for correct normalization in future selections
        min_max_stats.update(node.Q[action])


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

        policy_probs = policy_probs.squeeze(0).detach().cpu().numpy()
        
        # Add Dirichlet noise to root priors for exploration
        noise = np.random.dirichlet([dirichlet_alpha] * action_space_size)
        policy_probs = 0.75 * policy_probs + 0.25 * noise

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

    visit_counts = np.array(root.N, dtype=np.float32)
    
    if visit_counts.sum() > 0:
        if temperature == 0:
            policy_target = np.zeros_like(visit_counts)
            policy_target[np.argmax(visit_counts)] = 1.0
        else:
            # Temperature scaling
            visit_counts_temp = visit_counts ** (1 / temperature)
            policy_target = visit_counts_temp / visit_counts_temp.sum()
    else:
        policy_target = np.ones_like(visit_counts, dtype=np.float32) / len(visit_counts)

    # Choose action based on the policy target
    action = int(np.random.choice(len(policy_target), p=policy_target))
    
    return action, policy_target, float(v_root.item())