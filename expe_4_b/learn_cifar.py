"""
Details:
    * 64K iterations is 205 epochs,
    * 32K and 48K steps correspond to 103 and 154 epochs respectively.

The expected time on my laptop is of 2 minutes per epoch.
The total time should be 205*2 = 405 minutes per run,
so around 6.75 hours on my laptop.
Probably it is faster on the normal 
"""

import os
import torch
import json
import sys

import click
from tqdm import tqdm
import numpy as np
from dotenv import find_dotenv, load_dotenv
from datetime import datetime as dt

from custom_resnet import resnet56
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn

from torchvision.datasets import CIFAR10, CIFAR100

from utils import Logger, cifar_augmentation, cifar_normalization, accuracy


@click.command()
@click.option("--db_name", default="cifar10", help="Database name.")
@click.option("--model_save_name", default="cifar_from_scratch", help="Model to save.")
@click.option(
    "--tb_logs_name",
    default="tb_cifar_scratch",
    help="Folder in which to save tensorboard logs.",
)
@click.option("--learning_rate", default=0.1, help="Learning rate.")
@click.option("--weight_decay", default=1e-4, help="Weight decay.")
@click.option("--batch_size", default=128, help="Batch size.")
@click.option("--epochs", default=205, help="Number of epochs to train for.")
@click.option(
    "--batches_per_validation",
    default=200,
    help="Do a validation every X training batches.",
)
@click.option(
    "--training_stats_save_interval",
    default=250,
    help="Save loss, top1 and top5 accuracy to tensorboard every X iterations.",
)
def main(
    db_name,
    model_save_name,
    tb_logs_name,
    learning_rate,
    weight_decay,
    batch_size,
    epochs,
    batches_per_validation,
    training_stats_save_interval,
):
    # Load env variables for path etc.
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    model_save_path = "models"
    log_path = "logs"
    model_name = model_save_name
    script_parameters = locals()

    original_date = "{}_{}_{}_{}".format(
        dt.now().month, dt.now().day, dt.now().hour, dt.now().minute
    )
    tensorboard_log_dir = "{}/{}".format(tb_logs_name, original_date)
    # Write variables to tensorboard
    if not os.path.exists(os.path.join(log_path, tensorboard_log_dir)):
        os.makedirs(os.path.join(log_path, tensorboard_log_dir))
    with open(
        "{}/params.json".format(os.path.join(log_path, tensorboard_log_dir)), "wt+"
    ) as f:
        # The json dump does not work when debugging for some reason so we use
        # sys.gettrace to check if we're debugging
        if sys.gettrace() is None:
            json.dump(script_parameters, f, indent=2, sort_keys=True)

    loss_fn = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = resnet56(100).to(device)  # Number of classes = 100
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True,
    )

    num_workers = 16
    train_tr, test_tr = cifar_augmentation(), cifar_normalization()
    if db_name == "cifar10":
        cifar_train = CIFAR10("./data", download=True, train=True, transform=train_tr)
        cifar_val = CIFAR10("./data", download=True, train=False, transform=test_tr)
    elif db_name == "cifar100":
        cifar_train = CIFAR100("./data", download=True, train=True, transform=train_tr)
        cifar_val = CIFAR100("./data", download=True, train=False, transform=test_tr)

    train_loader = torch.utils.data.DataLoader(
        cifar_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        cifar_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    writer = SummaryWriter(log_dir=os.path.join(log_path, tensorboard_log_dir))
    train_logger = Logger(tensorboard_writer=writer)
    len_train_dataset = len(train_loader.dataset)
    len_val_dataset = len(val_loader.dataset)
    model.train()

    print("Starting training...")
    total_iteration = 0
    scheduler = MultiStepLR(optimizer, [103, 154], gamma=0.1)
    for epoch in range(epochs):
        train_progress = tqdm(
            train_loader,
            position=0,
            leave=True,
            ncols=100,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )
        train_progress.set_description("Train - epoch {}".format(epoch))
        train_logger.set_tqdm_logger(train_progress)
        i_img_train = 0
        for i, (images, labels) in enumerate(train_progress):
            # Training loop
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = loss_fn(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i_img_train += labels.shape[0]
            if i % training_stats_save_interval == 0:
                labels = labels.to(device)
                top1, top5 = accuracy(output, labels, [1, 5])
                # calculate the total iteration when counting all iterations over all epochs
                train_logger.log("train_loss", loss.item(), iteration=total_iteration)
                train_logger.log("train_top1", top1.item(), iteration=total_iteration)
                train_logger.log("train_top5", top5.item(), iteration=total_iteration)

            if i % batches_per_validation == 0:
                val_progress = tqdm(
                    val_loader,
                    position=0,
                    leave=True,
                )
                model.eval()
                with torch.no_grad():
                    sum_loss, sum_top1, sum_top5 = [0, 0, 0]
                    val_progress.set_description("Val - epoch {}".format(epoch))
                    i_img_val = 0

                    for val_images, val_labels in val_progress:
                        val_images = val_images.to(device)
                        val_labels = val_labels.to(device)

                        output = model(val_images)

                        loss = loss_fn(output, val_labels)

                        norm = torch.sum(torch.abs(output))

                        top1, top5 = accuracy(output, val_labels, (1, 5))
                        i_img_val += val_labels.shape[0]

                        # Need to multiply by batch size so the last (possibly smaller)
                        # batch does not skew results
                        current_batch_size = val_labels.shape[0]
                        sum_loss += loss * current_batch_size
                        sum_top1 += top1 * current_batch_size
                        sum_top5 += top5 * current_batch_size
                    val_dataset_size = len(val_loader.dataset)
                    avg_loss, avg_top1, avg_top5 = (
                        (sum_loss / val_dataset_size).item(),
                        (sum_top1 / val_dataset_size).item(),
                        (sum_top5 / val_dataset_size).item(),
                    )
                    train_logger.log("val_loss", avg_loss, iteration=total_iteration)
                    train_logger.log("val_top1", avg_top1, iteration=total_iteration)
                    train_logger.log("val_top5", avg_top5, iteration=total_iteration)
                model.train()
            total_iteration += 1

        if (epoch % 100) == 0:
            # Save intermediate checkpoints every two epochs
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(
                    model_save_path,
                    model_name + "-" + original_date + "-epoch{}.pt".format(epoch),
                ),
            )
        scheduler.step()

    # Save the checkpoint
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(
            model_save_path,
            model_name + "-" + original_date + ".pt",
        ),
    )
    # Save the performances of the model in a convenient file
    path_perfos = os.path.join(
        model_save_path, model_name + "-" + original_date + ".json"
    )

    d_perfos = {"avg_loss": avg_loss, "avg_top1": avg_top1, "avg_top5": avg_top5}
    with open(path_perfos, "wt") as f:
        json.dump(d_perfos, f)


if __name__ == "__main__":
    main()
