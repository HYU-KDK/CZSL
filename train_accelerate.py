import argparse
import os
import pickle
import pprint

import numpy as np
import torch
import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator

from datasets.composition_dataset import CompositionDataset
from datasets.read_datasets import DATASET_PATHS
from models.compositional_modules import get_model
from utils import set_seed

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def train_model(model, optimizer, train_dataset, config, accelerator):
    """Function to train the model to predict attributes with cross entropy loss.

    Args:
        model (nn.Module): the model to compute the similarity score with the images.
        optimizer (nn.optim): the optimizer with the learnable parameters.
        train_dataset (CompositionDataset): the train dataset
        config (argparse.ArgumentParser): the config
        accelerator (Accelerator): the accelerator instance

    Returns:
        tuple: the trained model (or the best model) and the optimizer
    """
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
    #
    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).to(accelerator.device)
    i = 0
    train_losses = []

    torch.autograd.set_detect_anomaly(True)

    for i in range(config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1),
            disable=not accelerator.is_local_main_process
        )

        epoch_train_losses = []
        for bid, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                batch_img, batch_target = batch[0], batch[3]
                batch_target = batch_target.to(accelerator.device)
                batch_img = batch_img.to(accelerator.device)
                
                unwrapped_model = accelerator.unwrap_model(model)
                batch_feat = unwrapped_model.encode_image(batch_img)

                logits = model(batch_feat, train_pairs)

                loss = loss_fn(logits, batch_target)

                # backward pass
                accelerator.backward(loss)

                # weights update
                optimizer.step()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            progress_bar.set_postfix(
                {"train loss": np.mean(epoch_train_losses[-50:])}
            )

            progress_bar.update()

        progress_bar.close()
        accelerator.print(
            f"epoch {i +1} train loss {np.mean(epoch_train_losses)}"
        )
        train_losses.append(np.mean(epoch_train_losses))

        if (i + 1) % config.save_every_n == 0:
            save_soft_embeddings(model, config, accelerator, epoch=i + 1)

    return model, optimizer


def save_soft_embeddings(model, config, accelerator, epoch=None):
    """Function to save soft embeddings.

    Args:
        model (nn.Module): the CSP/COOP module
        config (argparse.ArgumentParser): the config
        accelerator (Accelerator): the accelerator instance
        epoch (int, optional): epoch number for the soft embedding.
            Defaults to None.
    """
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
    parser.add_argument(
        "--experiment_name",
        help="name of the experiment",
        type=str,
    )
    parser.add_argument("--dataset", help="name of the dataset", type=str)
    parser.add_argument(
        "--lr", help="learning rate", type=float, default=5e-05
    )
    parser.add_argument(
        "--weight_decay", help="weight decay", type=float, default=1e-05
    )
    parser.add_argument(
        "--clip_model", help="clip model type", type=str, default="ViT-B/32"
    )
    parser.add_argument(
        "--epochs", help="number of epochs", default=20, type=int
    )
    parser.add_argument(
        "--train_batch_size", help="train batch size", default=64, type=int
    )
    parser.add_argument(
        "--eval_batch_size", help="eval batch size", default=1024, type=int
    )
    parser.add_argument(
        "--evaluate_only",
        help="directly evaluate on the dataset without any training",
        action="store_true",
    )
    parser.add_argument(
        "--context_length",
        help="sets the context length of the clip model",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--attr_dropout",
        help="add dropout to attributes",
        type=float,
        default=0.0,
    )
    parser.add_argument("--save_path", help="save path", type=str)
    parser.add_argument(
        "--save_every_n",
        default=1,
        type=int,
        help="saves the model every n epochs; "
        "this is useful for validation/grid search",
    )
    parser.add_argument(
        "--save_model",
        help="indicate if you want to save the model state dict()",
        action="store_true",
    )
    parser.add_argument("--seed", help="seed value", default=0, type=int)

    parser.add_argument(
        "--gradient_accumulation_steps",
        help="number of gradient accumulation steps",
        default=1,
        type=int
    )

    config = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)

    # set the seed value
    set_seed(config.seed)

    accelerator.print("training details")
    if accelerator.is_local_main_process:
        pprint.pprint(config)

    if os.path.exists(config.save_path):
        if accelerator.is_local_main_process:
            accelerator.print('file already exists')
            accelerator.print('exiting!')
        exit(0)

    # This should work for mit-states, ut-zappos, and maybe c-gqa.
    dataset_path = DATASET_PATHS[config.dataset]
    train_dataset = CompositionDataset(dataset_path,
                                       phase='train',
                                       split='compositional-split-natural')

    model, optimizer = get_model(train_dataset, config, accelerator.device)

    model = model.to(accelerator.device)

    # soft_embeddings가 파라미터가 아닌 일반 텐서로 정의되어 넘어가지 않았을 경우를 위한 강제 할당
    if hasattr(model, 'soft_embeddings'):
        if isinstance(model.soft_embeddings, torch.Tensor):
            model.soft_embeddings = model.soft_embeddings.to(accelerator.device)

    accelerator.print("model dtype", model.dtype)
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.print("soft embedding dtype", unwrapped_model.soft_embeddings.dtype)

    if not config.evaluate_only:
        model, optimizer = train_model(
            model,
            optimizer,
            train_dataset,
            config,
            accelerator,
        )

    save_soft_embeddings(
        model,
        config,
        accelerator,
    )

    if accelerator.is_local_main_process:
        with open(os.path.join(config.save_path, "config.pkl"), "wb") as fp:
            pickle.dump(config, fp)

        if config.save_model:
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(
                unwrapped_model.state_dict(),
                os.path.join(
                    config.save_path,
                    'final_model.pt'))

    accelerator.print("done!")
