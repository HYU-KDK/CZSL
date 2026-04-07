import json
import os
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.csp import csp_init, CSPInterface

class LLMClusProInterface(CSPInterface):
    def __init__(
        self,
        clip_model,
        config,
        offset,
        soft_embeddings,
        class_token_ids,
        device="cuda:0",
        attr_dropout=0.0,
        num_clusters=10,
        cluster_mapping=None,
        mu=0.99
    ):
        super().__init__(
            clip_model,
            config,
            offset,
            soft_embeddings,
            class_token_ids,
            device=device,
            enable_pos_emb=True,
            attr_dropout=attr_dropout
        )
        self.num_clusters = num_clusters
        self.mu = mu
        
        # Initialize prototypes P_k
        self.register_buffer("prototypes", torch.zeros(num_clusters, clip_model.text_projection.shape[1], device=device))
        
        # cluster_mapping should be dict: obj_idx -> cluster_idx
        self.cluster_mapping = cluster_mapping
        
        self.is_prototypes_initialized = False

    def init_prototypes(self, cluster_names, device):
        # Initialize prototypes using CLIP text embeddings of cluster names
        with torch.no_grad():
            tokenized = clip.tokenize([f"a photo of {name}" for name in cluster_names], context_length=self.config.context_length).to(device)
            text_features = self.clip_model.encode_text(tokenized)
            text_features = F.normalize(text_features, dim=-1)
            self.prototypes.copy_(text_features)
        self.is_prototypes_initialized = True

    def update_prototypes(self, img_features, obj_indices):
        # img_features: [B, D]
        # obj_indices: [B]
        if not self.is_prototypes_initialized:
            return
            
        with torch.no_grad():
            for k in range(self.num_clusters):
                # Find which images belong to cluster k
                mask = (self.cluster_mapping[obj_indices] == k)
                if mask.sum() > 0:
                    f_k_mean = img_features[mask].mean(dim=0)
                    f_k_mean = F.normalize(f_k_mean, dim=-1)
                    
                    new_p_k = self.mu * self.prototypes[k] + (1 - self.mu) * f_k_mean
                    self.prototypes[k].copy_(F.normalize(new_p_k, dim=-1))

    def get_all_object_features(self, all_obj_names, device):
        # Cache or compute text features for all objects using the current state
        # For simplicity, we just use the frozen CLIP text encoder on "a photo of [obj]"
        with torch.no_grad():
            tokenized = clip.tokenize([f"a photo of {name}" for name in all_obj_names], context_length=self.config.context_length).to(device)
            text_features = self.clip_model.encode_text(tokenized)
            text_features = F.normalize(text_features, dim=-1)
        return text_features

def get_llm_cluspro(train_dataset, config, device, llm_clusters_path="llm_clusters.json"):
    (
        clip_model,
        soft_embedding,
        class_token_ids,
        offset
    ) = csp_init(train_dataset, config, device)

    optimizer = torch.optim.Adam(
        [soft_embedding],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Load LLM clusters
    with open(llm_clusters_path, "r") as f:
        data = json.load(f)
    
    clusters = data["clusters"]
    obj_to_cluster = data["obj_to_cluster"]
    
    cluster_names = list(clusters.keys())
    num_clusters = len(cluster_names)
    
    # Map dataset obj_idx to cluster_idx
    obj_idx_to_cluster_idx = torch.zeros(len(train_dataset.objs), dtype=torch.long, device=device)
    for obj_name, info in obj_to_cluster.items():
        if obj_name in train_dataset.obj2idx:
            obj_idx = train_dataset.obj2idx[obj_name]
            obj_idx_to_cluster_idx[obj_idx] = info["cluster_idx"]

    interface = LLMClusProInterface(
        clip_model,
        config,
        offset,
        soft_embedding,
        class_token_ids,
        device,
        attr_dropout=config.attr_dropout,
        num_clusters=num_clusters,
        cluster_mapping=obj_idx_to_cluster_idx,
        mu=0.99
    )
    
    interface.init_prototypes(cluster_names, device)

    return interface, optimizer
