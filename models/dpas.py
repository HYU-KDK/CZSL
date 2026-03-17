import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from models.dpc_alpha import DPCAlphaInterface

# -------------------------------------------------------------------------
# CLUSPRO Helper Modules (Re-implemented for self-containment)
# -------------------------------------------------------------------------

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck=64, dropout=0.0, scale=0.1):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck)
        self.relu = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = scale
        
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5) if hasattr(math, 'sqrt') else 2.23)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True):
        down = self.dropout(self.relu(self.down_proj(x)))
        up = self.up_proj(down) * self.scale
        return up + x if add_residual else up

class Disentangler(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.fc = nn.Linear(emb_dim, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        # x: [B, D]
        x = F.relu(self.bn(self.fc(x)))
        x = F.dropout(x, p=0.1, training=self.training)
        return x

class MulitHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        _, M, _ = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.cross_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, q, kv):
        q = q + self.cross_attn(q, kv, kv)
        q = q + self.dropout(self.mlp(self.norm(q)))
        return q

import math

class CustomTextEncoder(torch.nn.Module):
    def __init__(self, clip_model, dtype=torch.float16):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = dtype

    def forward(self, token_ids, token_tensors=None, enable_pos_emb=True):
        if token_tensors is not None:
            text_features = token_tensors
        else:
            text_features = self.clip_model.token_embedding(token_ids)

        text_features = text_features.type(self.dtype)
        x = text_features + self.clip_model.positional_embedding.type(self.dtype) if enable_pos_emb else text_features
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        if self.training:
            x = checkpoint(self.clip_model.transformer, x, use_reentrant=False)
        else:
            x = self.clip_model.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x)
        tf = x[torch.arange(x.shape[0]), token_ids.argmax(dim=-1)] @ self.clip_model.text_projection
        return tf

# -------------------------------------------------------------------------
# DPAS Interface with Multi-Prototype and Disentanglement
# -------------------------------------------------------------------------

class DPASInterface(DPCAlphaInterface):
    def __init__(
        self,
        clip_model,
        config,
        offset,
        soft_embeddings_primitive,
        soft_embeddings_contextual,
        class_token_ids,
        device="cuda:0",
        enable_pos_emb=True,
        attr_dropout=0.0,
        feature_layer=6,
        num_prototypes=1,
    ):
        super().__init__(
            clip_model,
            config,
            offset,
            soft_embeddings_primitive,
            soft_embeddings_contextual,
            class_token_ids,
            device=device,
            enable_pos_emb=enable_pos_emb,
            attr_dropout=attr_dropout,
        )

        self.feature_layer = feature_layer
        self.num_prototypes = num_prototypes
        self._local_feat_cache = {}
        
        vision_width = clip_model.visual.class_embedding.shape[0]
        embed_dim = clip_model.text_projection.shape[1]

        # CLUSPRO Components - Must match CLIP's dtype (Half)
        layers = clip_model.visual.transformer.layers
        self.vision_adapters = nn.ModuleList([
            Adapter(vision_width, bottleneck=64, dropout=0.1) for _ in range(2 * layers)
        ]).to(device).type(clip_model.dtype)

        self.attr_disentangler = Disentangler(embed_dim).to(device).type(clip_model.dtype)
        self.obj_disentangler = Disentangler(embed_dim).to(device).type(clip_model.dtype)
        self.attr_refiner = Disentangler(embed_dim).to(device).type(clip_model.dtype)
        self.obj_refiner = Disentangler(embed_dim).to(device).type(clip_model.dtype)

        self.cmt = nn.ModuleList([
            CrossAttentionLayer(embed_dim, nhead=8, dropout=0.1)
            for _ in range(1)
        ]).to(device).type(clip_model.dtype)

        self.cmt_lambda = nn.Parameter(torch.ones(embed_dim, device=device).type(clip_model.dtype) * 0.1)
        self.patch_norm = nn.LayerNorm(embed_dim).to(device).type(clip_model.dtype)

        # DPAS Components - Keep as Float32 for stability, cast inputs in forward
        self.prompt_shifter = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
        ).to(device)

        # Predictor Input: norm_entropy(1) + norm_max_logit(1) + f_global(D) + shift_norm(1) + f_local_norm(1)
        self.advanced_alpha_predictor = nn.Sequential(
            nn.Linear(3 + embed_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(device)

        self.text_encoder_wrapped = CustomTextEncoder(clip_model, clip_model.dtype)
        self._register_local_hook()

    def _register_local_hook(self):
        def hook_fn(module, inp, out):
            self._local_feat_cache['local_cls'] = out[0].detach()
            self._local_feat_cache['local_patches'] = out[1:].detach()

        try:
            block = self.clip_model.visual.transformer.resblocks[self.feature_layer]
            self._hook_handle = block.register_forward_hook(hook_fn)
        except Exception as e:
            self._hook_handle = None

    def encode_image_with_adapters(self, x):
        visual = self.clip_model.visual
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)
        layers = visual.transformer.layers
        
        def run_block(x_in, block_idx):
            block = visual.transformer.resblocks[block_idx]
            # MHA with adapter
            res = x_in
            x_mha = block.attention(block.ln_1(x_in))
            x_mha = x_mha + self.vision_adapters[block_idx](x_mha, add_residual=False) + res

            # FFN with adapter
            res = x_mha
            x_ffn = block.mlp(block.ln_2(x_mha))
            x_ffn = x_ffn + self.vision_adapters[block_idx + layers](x_ffn, add_residual=False) + res
            return x_ffn

        for i in range(layers):
            if self.training:
                x = checkpoint(run_block, x, i, use_reentrant=False)
            else:
                x = run_block(x, i)
        
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x)
        
        if visual.proj is not None:
            f_global = x[:, 0, :] @ visual.proj
            f_patches = x[:, 1:, :] @ visual.proj
        else:
            f_global, f_patches = x[:, 0, :], x[:, 1:, :]
        return f_global, f_patches

    def forward(self, imgs, pair_idx):
        batch_size, n_pairs = imgs.shape[0], pair_idx.shape[0]
        imgs = imgs.to(self.device).type(self.clip_model.dtype)

        f_global, f_patches = self.encode_image_with_adapters(imgs)
        f_mid_patches = self._local_feat_cache.get('local_patches')
        f_local_signal = f_mid_patches.mean(dim=0) if f_mid_patches is not None else self._local_feat_cache.get('local_cls', f_global)

        # 1. Primitive path (CLUSPRO Multi-Prototype + CMT) - Memory Optimized
        token_tensor_prim, _ = self._construct_multi_prototype_tokens(pair_idx)
        total_p_combos = token_tensor_prim.shape[0] # N_pairs * K*K
        
        prim_feats_base_list = []
        ids_prim = self.token_ids[0].unsqueeze(0).expand(total_p_combos, -1).to(self.device)
        
        torch.cuda.empty_cache()
        for i in range(0, total_p_combos, 512):
            feat_chunk = self.text_encoder_wrapped(ids_prim[i:i+512], token_tensor_prim[i:i+512])
            prim_feats_base_list.append(feat_chunk)
        
        prim_feats_base = torch.cat(prim_feats_base_list, dim=0) # [N_pairs * K*K, D]
        f_patches_norm = self.patch_norm(f_patches) # [B, S-1, D]
        f_global_norm = f_global / f_global.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        
        # Process pairs in chunks for CMT refinement to avoid OOM
        logits_prim_chunks = []
        pair_chunk_sz = 128
        K2 = self.num_prototypes ** 2
        for i in range(0, n_pairs, pair_chunk_sz):
            end_p = min(i + pair_chunk_sz, n_pairs)
            chunk_base = prim_feats_base[i*K2 : end_p*K2] # [Chunk*K2, D]
            
            # Expand and refine
            t_cmt = chunk_base.unsqueeze(0).expand(batch_size, -1, -1)
            for layer in self.cmt:
                t_cmt = layer(t_cmt, f_patches_norm)
            
            t_ref = chunk_base.unsqueeze(0) + self.cmt_lambda.to(t_cmt.dtype) * t_cmt
            t_ref = t_ref / t_ref.norm(dim=-1, keepdim=True)
            
            # Logits for this chunk: [Batch, Chunk*K2]
            l_p_chunk = logit_scale * (f_global_norm.unsqueeze(1) @ t_ref.transpose(1, 2)).squeeze(1)
            # Max over prototypes: [Batch, Chunk]
            l_p_chunk_max, _ = l_p_chunk.view(batch_size, -1, K2).max(dim=-1)
            logits_prim_chunks.append(l_p_chunk_max)
            
        logits_prim = torch.cat(logits_prim_chunks, dim=1) # [B, N_pairs]

        # 2. Contextual path (DPAS Shifter) - Feature-Level Shifting (Memory Efficient)
        shift = self.prompt_shifter(f_local_signal.float()).to(self.clip_model.dtype) # [B, D]
        
        ids_single = self.token_ids[0].to(self.device).unsqueeze(0)
        base_t = self.clip_model.token_embedding(ids_single.expand(n_pairs, -1)).type(self.clip_model.dtype)
        eos_idx = int(ids_single[0].argmax())
        ctx_embs = self.attr_dropout(self.soft_embeddings_contextual).type(self.clip_model.dtype)
        base_t[:, eos_idx - 2, :] = ctx_embs[pair_idx[:, 0]]
        base_t[:, eos_idx - 1, :] = ctx_embs[pair_idx[:, 1] + self.offset]
        
        ctx_feats_base_list = []
        for i in range(0, n_pairs, 1024):
            f_chunk = self.text_encoder_wrapped(ids_single.expand(min(1024, n_pairs - i), -1), base_t[i:i+1024])
            ctx_feats_base_list.append(f_chunk)
        
        ctx_feats_base = torch.cat(ctx_feats_base_list, dim=0) # [N_pairs, D]
        
        # Apply shift to features: [B, N_pairs, D]
        ctx_feats = ctx_feats_base.unsqueeze(0) + shift.unsqueeze(1)
        ctx_feats = ctx_feats / ctx_feats.norm(dim=-1, keepdim=True)
        
        logits_ctx = logit_scale * (f_global_norm.unsqueeze(1) @ ctx_feats.transpose(1, 2)).squeeze(1)

        # 3. Gating
        logits_p_f32 = logits_prim.float()
        norm_entropy = (-(F.softmax(logits_p_f32, dim=-1) * F.log_softmax(logits_p_f32, dim=-1)).sum(dim=-1, keepdim=True)) / torch.log(torch.tensor(float(n_pairs), device=self.device))
        norm_max_l = logits_p_f32.max(dim=-1, keepdim=True)[0] / logit_scale
        
        alpha_input = torch.cat([
            norm_entropy, 
            norm_max_l, 
            f_global_norm.float(), 
            shift.norm(dim=-1, keepdim=True).float(), 
            f_local_signal.float().norm(dim=-1, keepdim=True)
        ], dim=-1)
        
        alpha = torch.sigmoid(self.gating_param.float() + self.advanced_alpha_predictor(alpha_input)).to(self.clip_model.dtype)
        
        return (alpha * logits_prim + (1.0 - alpha) * logits_ctx) / self.temp

    def _construct_multi_prototype_tokens(self, pair_idx):
        n_pairs = pair_idx.shape[0]
        K = self.num_prototypes
        prim_embs = self.soft_embeddings_primitive.view(-1, K, self.soft_embeddings_primitive.shape[-1])
        a_embs = prim_embs[pair_idx[:, 0]].unsqueeze(2).repeat(1, 1, K, 1).view(-1, prim_embs.shape[-1])
        o_embs = prim_embs[pair_idx[:, 1] + self.offset].unsqueeze(1).repeat(1, K, 1, 1).view(-1, prim_embs.shape[-1])
        
        total = n_pairs * K * K
        ids = self.token_ids[0].unsqueeze(0).repeat(total, 1).to(self.device)
        base = self.clip_model.token_embedding(ids).type(self.clip_model.dtype)
        eos = int(self.token_ids[0].argmax())
        base[:, eos - 2, :] = self.attr_dropout(a_embs).type(self.clip_model.dtype)
        base[:, eos - 1, :] = self.attr_dropout(o_embs).type(self.clip_model.dtype)
        return base, None

def get_dpas(train_dataset, config, device):
    from models.dpc import dpc_init
    clip_model, soft_prim, soft_ctx, token_ids, offset = dpc_init(train_dataset, config, device, num_prototypes=getattr(config, 'num_prototypes', 1))
    
    # Explicitly freeze clip_model base parameters to save massive VRAM
    for param in clip_model.parameters():
        param.requires_grad = False
    
    interface = DPASInterface(clip_model, config, offset, soft_prim, soft_ctx, token_ids, device=device, num_prototypes=getattr(config, 'num_prototypes', 1))
    
    params = [{'params': [soft_prim, soft_ctx]}]
    params.append({
        'params': (list(interface.vision_adapters.parameters()) + list(interface.attr_disentangler.parameters()) + list(interface.obj_disentangler.parameters()) + 
                   list(interface.attr_refiner.parameters()) + list(interface.obj_refiner.parameters()) + list(interface.cmt.parameters()) + 
                   [interface.cmt_lambda] + list(interface.prompt_shifter.parameters()) + list(interface.advanced_alpha_predictor.parameters()) + [interface.gating_param]),
        'lr': 0.001
    })
    return interface, torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
