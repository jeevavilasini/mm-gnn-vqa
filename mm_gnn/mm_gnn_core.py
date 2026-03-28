import torch
import torch.nn as nn
from mm_gnn.aggregators import VisualSemanticAggregator, SemanticSemanticAggregator, SemanticNumericAggregator

class AnswerPredictor(nn.Module):
    """
    Implements the Answer Prediction module (Section 3.3 of the paper).
    Uses an attention mechanism to select the most relevant OCR token.
    """
    
    def __init__(self, visual_dim=2048, semantic_dim=1200, q_dim=512):
        super().__init__()
        # Simplified attention layers to score OCR tokens against the question
        self.attn_c = nn.Linear(semantic_dim + q_dim, 1)
        
    def forward(self, c_nodes, q_feat):
        # c_nodes are the fully updated OCR tokens
        num_tokens = c_nodes.size(0)
        
        # Expand question feature to match number of tokens
        q_expanded = q_feat.unsqueeze(0).expand(num_tokens, -1)
        
        # Combine token features with question to see which token answers the question
        combined = torch.cat([c_nodes, q_expanded], dim=1)
        
        # Calculate a score for each OCR token
        scores = self.attn_c(combined).squeeze(1)
        
        # Find the index of the highest scoring token
        best_idx = torch.argmax(scores).item()
        
        return best_idx

class MMGNN_Pipeline(nn.Module):
    """
    The master Multi-Modal Graph Neural Network pipeline.
    """
    def __init__(self):
        super().__init__()
        print("Initializing MM-GNN Pipeline...")
        # Initialize the 3 Aggregators
        self.vs_aggregator = VisualSemanticAggregator()
        self.ss_aggregator = SemanticSemanticAggregator()
        self.sn_aggregator = SemanticNumericAggregator()
        
        # Initialize the Answer Predictor
        self.predictor = AnswerPredictor()

    def forward(self, extracted_data):
        # 1. Unpack the extracted nodes and boxes
        v_nodes = extracted_data["visual"]
        s_nodes = extracted_data["semantic"]
        n_nodes = extracted_data["numeric"]
        v_bboxes = extracted_data["v_bboxes"]
        s_bboxes = extracted_data["s_bboxes"]
        q_feat = extracted_data["question"]
        raw_texts = extracted_data["raw_texts"]

        # 2. Step 1: Visual-Semantic Aggregation
        v_updated, s_updated_1 = self.vs_aggregator(v_nodes, s_nodes, v_bboxes, s_bboxes, q_feat)
        
        # 3. Step 2: Semantic-Semantic Aggregation
        s_updated_2 = self.ss_aggregator(s_updated_1, s_bboxes, q_feat)
        
        # 4. Step 3: Semantic-Numeric Aggregation 
        # (For this baseline, we append 0s or use baseline numeric aggregator)
        # We simulate the final OCR token representation 'c' mentioned in Sec 3.2
        c_nodes = self.sn_aggregator(n_nodes, s_updated_2)
        if c_nodes is None:
            # Fallback if no numeric nodes exist: pad semantic nodes to 900 dimensions
            padding = torch.zeros((s_updated_2.size(0), 300))
            c_nodes = torch.cat([s_updated_2, padding], dim=1)

        # 5. Answer Prediction (Copy Mechanism)
        best_token_idx = self.predictor(c_nodes, q_feat)
        
        # 6. Retrieve the actual text word corresponding to the best token
        predicted_answer = raw_texts[best_token_idx]
        
        return predicted_answer