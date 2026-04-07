import argparse
import os
import pickle
import pprint

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator

from datasets.composition_dataset import CompositionDataset
from datasets.read_datasets import DATASET_PATHS
from models.llm_cluspro import get_llm_cluspro
from utils.core import set_seed

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def train_model(model, optimizer, train_dataset, config, accelerator):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True
    )

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    model.train()
    loss_fn = CrossEntropyLoss()
    
    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx
    
    all_obj_names = train_dataset.objs

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).to(accelerator.device)

    # Pre-compute or get all object features for Phase 3
    # We do it once per epoch to save time, or dynamically
    unwrapped_model = accelerator.unwrap_model(model)
    t_objs = unwrapped_model.get_all_object_features(all_obj_names, accelerator.device)
    
    # Pre-compute cluster assignments for all objects
    cluster_mapping = unwrapped_model.cluster_mapping # Shape [num_objs]

    torch.autograd.set_detect_anomaly(True)
    
    tau = config.tau
    alpha = config.alpha
    beta = config.beta

    for epoch in range(config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (epoch + 1),
            disable=not accelerator.is_local_main_process
        )

        epoch_train_losses = []
        epoch_base_losses = []
        epoch_inter_losses = []
        epoch_intra_losses = []
        
        # Update text features if they change over time (e.g. if we update soft embeddings)
        if config.update_t_objs_every_epoch:
            unwrapped_model = accelerator.unwrap_model(model)
            t_objs = unwrapped_model.get_all_object_features(all_obj_names, accelerator.device)

        for bid, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                batch_img, _, batch_obj, batch_target = batch[0], batch[1], batch[2], batch[3]
                batch_target = batch_target.to(accelerator.device)
                batch_img = batch_img.to(accelerator.device)
                batch_obj = batch_obj.to(accelerator.device)
                
                # Forward pass: extract features
                unwrapped_model = accelerator.unwrap_model(model)
                batch_feat = unwrapped_model.encode_image(batch_img)
                
                # Base Loss: Phase 0
                logits = model(batch_feat, train_pairs)
                loss_base = loss_fn(logits, batch_target)

                # --- Phase 2: Inter-Cluster Loss ---
                # $L_{inter} = - 1/|B| \sum \log( \frac{\exp(f_x \cdot P_k / \tau)}{\sum_j \exp(f_x \cdot P_j / \tau)} )$
                
                normalized_feat = F.normalize(batch_feat, dim=-1)
                # P_k: model.prototypes -> [K, D]
                # sim_inter: [B, K]
                sim_inter = torch.matmul(normalized_feat, unwrapped_model.prototypes.t()) / tau
                
                # Target cluster indices for each image in batch
                target_clusters = cluster_mapping[batch_obj]
                loss_inter = loss_fn(sim_inter, target_clusters)

                # --- Phase 3: Intra-Cluster Hard Negative Loss ---
                # $L_{intra} = - 1/|B| \sum \log( \frac{\exp(f_x \cdot t_{obj+} / \tau)}{... + \sum_{o^- \in S_k} \exp(f_x \cdot t_{o^-} / \tau)} )$
                
                # t_objs is [num_objs, D]
                # normalized_feat is [B, D]
                # sim_intra: [B, num_objs]
                sim_intra = torch.matmul(normalized_feat, t_objs.t()) / tau
                
                # Mask out objects that are NOT in the same cluster
                # For each item b in batch, its cluster is target_clusters[b]
                # Mask should be [B, num_objs], where mask[b, o] = 1 if cluster_mapping[o] == target_clusters[b]
                
                same_cluster_mask = (cluster_mapping.unsqueeze(0) == target_clusters.unsqueeze(1)) # [B, num_objs]
                
                # To compute softmax only over same cluster, we set the similarities of different clusters to -infinity
                sim_intra = sim_intra.masked_fill(~same_cluster_mask, -1e9)
                loss_intra = loss_fn(sim_intra, batch_obj)

                # --- Total Loss ---
                loss = loss_base + alpha * loss_inter + beta * loss_intra

                # backward pass
                accelerator.backward(loss)

                # weights update
                optimizer.step()
                optimizer.zero_grad()
                
                # Phase 1: Update Prototypes online (EMA)
                unwrapped_model.update_prototypes(batch_feat.detach(), batch_obj)

            epoch_train_losses.append(loss.item())
            epoch_base_losses.append(loss_base.item())
            epoch_inter_losses.append(loss_inter.item())
            epoch_intra_losses.append(loss_intra.item())
            
            progress_bar.set_postfix(
                {
                    "total": np.mean(epoch_train_losses[-50:]),
                    "base": np.mean(epoch_base_losses[-50:]),
                    "inter": np.mean(epoch_inter_losses[-50:]),
                    "intra": np.mean(epoch_intra_losses[-50:])
                }
            )
            progress_bar.update()

        progress_bar.close()
        accelerator.print(
            f"epoch {epoch +1} train loss {np.mean(epoch_train_losses):.4f} "
            f"(base: {np.mean(epoch_base_losses):.4f}, "
            f"inter: {np.mean(epoch_inter_losses):.4f}, "
            f"intra: {np.mean(epoch_intra_losses):.4f})"
        )

        if (epoch + 1) % config.save_every_n == 0:
            save_soft_embeddings(model, config, accelerator, epoch=epoch + 1)

    return model, optimizer


def save_soft_embeddings(model, config, accelerator, epoch=None):
    if not accelerator.is_local_main_process:
        return

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    # save the soft embedding
    with torch.no_grad():
        unwrapped_model = accelerator.unwrap_model(model)
        if epoch:
            soft_emb_path = os.path.join(
                config.save_path, f"soft_embeddings_epoch_{epoch}.pt"
            )
        else:
            soft_emb_path = os.path.join(
                config.save_path, "soft_embeddings.pt"
            )

        torch.save({"soft_embeddings": unwrapped_model.soft_embeddings}, soft_emb_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", help="name of the experiment", type=str, default="llm_cluspro")
    parser.add_argument("--dataset", help="name of the dataset", type=str, default="mit-states")
    parser.add_argument("--lr", help="learning rate", type=float, default=5e-05)
    parser.add_argument("--weight_decay", help="weight decay", type=float, default=1e-05)
    parser.add_argument("--clip_model", help="clip model type", type=str, default="ViT-B/16")
    parser.add_argument("--epochs", help="number of epochs", default=20, type=int)
    parser.add_argument("--train_batch_size", help="train batch size", default=64, type=int)
    parser.add_argument("--eval_batch_size", help="eval batch size", default=1024, type=int)
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--context_length", default=32, type=int)
    parser.add_argument("--attr_dropout", type=float, default=0.0)
    parser.add_argument("--save_path", help="save path", type=str, default="checkpoints/llm_cluspro")
    parser.add_argument("--save_every_n", default=1, type=int)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    
    # CLUSPRO specific arguments
    parser.add_argument("--llm_clusters_path", type=str, default="llm_clusters.json")
    parser.add_argument("--alpha", help="weight for inter-cluster loss", type=float, default=0.2)
    parser.add_argument("--beta", help="weight for intra-cluster loss", type=float, default=0.5)
    parser.add_argument("--tau", help="temperature for contrastive loss", type=float, default=0.1)
    parser.add_argument("--update_t_objs_every_epoch", action="store_true")

    config = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)

    set_seed(config.seed)

    accelerator.print("training details")
    if accelerator.is_local_main_process:
        pprint.pprint(config)

    if os.path.exists(config.save_path) and not config.evaluate_only:
        if accelerator.is_local_main_process:
            accelerator.print('file already exists')
            # accelerator.print('exiting!')
            # exit(0)
    
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path, exist_ok=True)

    dataset_path = DATASET_PATHS[config.dataset]
    train_dataset = CompositionDataset(dataset_path, phase='train', split='compositional-split-natural')

    model, optimizer = get_llm_cluspro(train_dataset, config, accelerator.device, llm_clusters_path=config.llm_clusters_path)
    model = model.to(accelerator.device) 

    if hasattr(model, 'soft_embeddings'):
        if isinstance(model.soft_embeddings, torch.Tensor):
            model.soft_embeddings = model.soft_embeddings.to(accelerator.device)

    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.print("model dtype", unwrapped_model.clip_model.dtype)
    accelerator.print("soft embedding dtype", unwrapped_model.soft_embeddings.dtype)

    if not config.evaluate_only:
        model, optimizer = train_model(
            model,
            optimizer,
            train_dataset,
            config,
            accelerator,
        )

        save_soft_embeddings(model, config, accelerator)

        if accelerator.is_local_main_process:
            with open(os.path.join(config.save_path, "config.pkl"), "wb") as fp:
                pickle.dump(config, fp)

            if config.save_model:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(
                    unwrapped_model.state_dict(),
                    os.path.join(config.save_path, 'final_model.pt')
                )

    accelerator.print("done!")
