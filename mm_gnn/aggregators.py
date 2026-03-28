import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualSemanticAggregator(nn.Module):
    def __init__(self, visual_dim=2048, semantic_dim=300, bbox_dim=10, q_dim=512):
        super().__init__()
        # MLPs for calculating attention scores
        self.f_s = nn.Linear(semantic_dim + bbox_dim, 512)
        self.f_v = nn.Linear(visual_dim + bbox_dim, 512)
        self.f_q = nn.Linear(q_dim, 512)

        # MLPs to encode features before passing them
        self.f_v_prime = nn.Linear(visual_dim, semantic_dim)
        self.f_s_prime = nn.Linear(semantic_dim, visual_dim)

    def forward(self, v_nodes, s_nodes, v_bboxes, s_bboxes, q_feat):
        # A. Combine node features with their location (bounding box)
        s_combined = torch.cat([s_nodes, s_bboxes], dim=1)
        v_combined = torch.cat([v_nodes, v_bboxes], dim=1)

        # B. Project to a common 512-dimensional space
        s_proj = self.f_s(s_combined)
        v_proj = self.f_v(v_combined)
        q_proj = self.f_q(q_feat)

        # C. Calculate Attention Scores
        v_q_combined = v_proj * q_proj.unsqueeze(0)
        attention_scores = torch.matmul(s_proj, v_q_combined.T)
        
        # D. Aggregate visual info and append it to the Semantic Node
        attention_weights_s = torch.softmax(attention_scores, dim=1)
        v_encoded = self.f_v_prime(v_nodes)
        aggregated_visual = torch.matmul(attention_weights_s, v_encoded)
        s_updated = torch.cat([s_nodes, aggregated_visual], dim=1)

        # E. Aggregate semantic info and append to Visual Node
        attention_weights_v = torch.softmax(attention_scores.T, dim=1)
        s_encoded = self.f_s_prime(s_nodes)
        aggregated_semantic = torch.matmul(attention_weights_v, s_encoded)
        v_updated = torch.cat([v_nodes, aggregated_semantic], dim=1)

        return v_updated, s_updated


class SemanticSemanticAggregator(nn.Module):
    # Note: semantic_dim is now 600 because it absorbed 300 visual features
    def __init__(self, semantic_dim=600, bbox_dim=10, q_dim=512):
        super().__init__()
        self.g_s1 = nn.Linear(semantic_dim + bbox_dim, 512)
        self.g_s2 = nn.Linear(semantic_dim + bbox_dim, 512)
        self.g_q = nn.Linear(q_dim, 512)
        self.g_s3 = nn.Linear(semantic_dim, 300)

    def forward(self, s_nodes, s_bboxes, q_feat):
        s_combined = torch.cat([s_nodes, s_bboxes], dim=1)
        
        s1_proj = self.g_s1(s_combined)
        s2_proj = self.g_s2(s_combined)
        q_proj = self.g_q(q_feat)
        
        s2_q_combined = s2_proj * q_proj.unsqueeze(0)
        attention_scores = torch.matmul(s1_proj, s2_q_combined.T)
        
        # Prevent self-attention (masking the diagonal)
        mask = torch.eye(attention_scores.size(0), device=attention_scores.device).bool()
        attention_scores.masked_fill_(mask, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        s_encoded = self.g_s3(s_nodes)
        aggregated_semantic = torch.matmul(attention_weights, s_encoded)
        s_updated = torch.cat([s_nodes, aggregated_semantic], dim=1)
        
        return s_updated


class SemanticNumericAggregator(nn.Module):
    # Note: semantic_dim is now 900
    def __init__(self, numeric_dim=1, semantic_dim=900, q_dim=512):
        super().__init__()
        # Simplified for baseline: encoding semantic context to pass to numeric nodes
        self.h = nn.Linear(semantic_dim, numeric_dim)
        
    def forward(self, n_nodes, s_nodes):
        if n_nodes is None or len(n_nodes) == 0:
            return n_nodes
            
        # Simplistic attention-less aggregation for baseline numeric update
        s_encoded = self.h(s_nodes)
        aggregated_semantic = torch.mean(s_encoded, dim=0, keepdim=True)
        aggregated_semantic = aggregated_semantic.repeat(n_nodes.size(0), 1)
        
        n_updated = torch.cat([n_nodes, aggregated_semantic], dim=1)
        return n_updated