import argparse
import os
import shutil


from dataset.dataset import MyDataset
import torch
import numpy as np
import random
from metrics.metric_tool import ConfuseMatrixMeter
from models.change_classifier import ChangeClassifier as Model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt


def parse_arguments():
    # Argument Parser creation
    parser = argparse.ArgumentParser(
        description="Parameter for data analysis, data cleaning and model training."
    )
    parser.add_argument(
        "--datapath",
        type=str,
        help="data path",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        help="log path",
    )

    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')

    parsed_arguments = parser.parse_args()

    # create log dir if it doesn't exists
    if not os.path.exists(parsed_arguments.log_path):
        os.mkdir(parsed_arguments.log_path)

    dir_run = sorted(
        [
            filename
            for filename in os.listdir(parsed_arguments.log_path)
            if filename.startswith("run_")
        ]
    )

    if len(dir_run) > 0:
        num_run = int(dir_run[-1].split("_")[-1]) + 1
    else:
        num_run = 0
    parsed_arguments.log_path = os.path.join(
        parsed_arguments.log_path, "run_%04d" % num_run + "/"
    )

    return parsed_arguments


def train(
    dataset_train,
    dataset_val,
    model,
    criterion,
    optimizer,
    scheduler,
    logpath,
    writer,
    epochs,
    save_after,
    device
):

    model = model.to(device)

    train_image_list = []
    val_image_list = []

    tool4metric = ConfuseMatrixMeter(n_class=2)

    def evaluate(reference, mask, image_list):
        # All the tensors on the device:
        reference = reference.to(device).float()
        mask = mask.to(device).float()

        # Evaluating the model:
        generated_mask = model(reference).squeeze(1)

        # Loss gradient descend step:
        it_loss = criterion(generated_mask, mask)

        # Feeding the comparison metric tool:
        bin_genmask = (generated_mask.to("cpu") >
                       0.5).detach().numpy().astype(int)
        mask = mask.to("cpu").numpy().astype(int)
        tool4metric.update_cm(pr=bin_genmask, gt=mask)

        # image_list.append([reference, mask, bin_genmask])

        return it_loss

    def training_phase(epc):
        tool4metric.clear()
        print("Epoch {}".format(epc))
        model.train()
        epoch_loss = 0.0
        for data in dataset_train:
            reference = data["image"]
            mask = data["mask"]
            # Reset the gradients:
            optimizer.zero_grad()

            # Loss gradient descend step:
            it_loss = evaluate(reference, mask, train_image_list)
            it_loss.backward()
            optimizer.step()

            # Track metrics:
            epoch_loss += it_loss.to("cpu").detach().numpy()
            ### end of iteration for epoch ###

        epoch_loss /= len(dataset_train)

        #########
        print("Training phase summary")
        print("Loss for epoch {} is {}".format(epc, epoch_loss))
        writer.add_scalar("Loss/epoch", epoch_loss, epc)
        scores_dictionary = tool4metric.get_scores()
        writer.add_scalar("IoU class change/epoch",
                          scores_dictionary["iou_1"], epc)
        writer.add_scalar("F1 class change/epoch",
                          scores_dictionary["F1_1"], epc)
        print(
            "IoU class change for epoch {} is {}".format(
                epc, scores_dictionary["iou_1"]
            )
        )
        print(
            "F1 class change for epoch {} is {}".format(
                epc, scores_dictionary["F1_1"])
        )
        print()
        writer.flush()

        ### Save the model ###
        if epc % save_after == 0:
            torch.save(
                model.state_dict(), os.path.join(logpath, "model_{}.pth".format(epc))
            )

    def validation_phase(epc):
        model.eval()
        epoch_loss_eval = 0.0
        tool4metric.clear()
        with torch.no_grad():
            for data in dataset_val:
                reference = data["image"]
                mask = data["mask"]
                img_name = data["img_name"]
                epoch_loss_eval += evaluate(reference, mask, val_image_list).to("cpu").numpy()

        epoch_loss_eval /= len(dataset_val)
        print("Validation phase summary")
        print("Loss for epoch {} is {}".format(epc, epoch_loss_eval))
        writer.add_scalar("Loss_val/epoch", epoch_loss_eval, epc)
        scores_dictionary = tool4metric.get_scores()
        writer.add_scalar("IoU_val class change/epoch",
                          scores_dictionary["iou_1"], epc)
        writer.add_scalar("F1_val class change/epoch",
                          scores_dictionary["F1_1"], epc)
        print(
            "IoU class change for epoch {} is {}".format(
                epc, scores_dictionary["iou_1"]
            )
        )
        print(
            "F1 class change for epoch {} is {}".format(
                epc, scores_dictionary["F1_1"])
        )
        print()

    for epc in range(epochs):
        training_phase(epc)
        validation_phase(epc)
        # scheduler step
        scheduler.step()
    #output_train_image(train_image_list, "/home/ramat/experiments/exp_tinyCD/exp136/train_image")
    #output_train_image(val_image_list, "/home/ramat/experiments/exp_tinyCD/exp136/val_image")
        
def output_train_image(
        image_list,
        save_path
):
    for k in range(len(image_list)):
        columns = 3
        rows = 3
        j = 0
        fig = plt.figure(figsize=(12, 16))
        for i in range(columns * rows)[::3]:
            if j < len(image_list[k]):
                img = image_list[k][0][j,0,:,:].squeeze().cpu()
                col1 = fig.add_subplot(rows, columns, i + 1)
                plt.imshow(img, cmap="gray")
                col2 = fig.add_subplot(rows, columns, i + 2)
                plt.imshow(image_list[k][1][j], cmap="gray")
                col3 = fig.add_subplot(rows, columns, i + 3)
                plt.imshow(image_list[k][2][j], cmap="gray")
                j += 1
                if i == 0:
                    col1.title.set_text("Data")
                    col2.title.set_text("Ground Truth")
                    col3.title.set_text("Prediction")
        plt.suptitle(f"Predictions", fontsize=16)
        plt_out_pth = os.path.join(save_path, f"prediction_image_{k}.png")
        plt.savefig(plt_out_pth, dpi=300)


def output_test_image(
    dataset_test,
    model,
    device,
    logpath
):
    model = model.to(device)
    model.eval()
    k = 0
    with torch.no_grad():
        for data in dataset_test:
            reference = data["image"]
            mask = data["mask"]
            # All the tensors on the device:
            reference = reference.to(device).float()
            mask = mask.to(device).float()

            # Evaluating the model:
            generated_mask = model(reference).squeeze(1)

            # Binarize mask
            bin_genmask = (generated_mask.to("cpu") >
                        0.5).detach().numpy().astype(int)
            mask = mask.to("cpu").numpy().astype(int)
            columns = 3
            rows = 3
            j = 0
            fig = plt.figure(figsize=(12, 16))
            for i in range(columns * rows)[::3]:
                img = reference[j,0,:,:].squeeze().cpu()
                col1 = fig.add_subplot(rows, columns, i + 1)
                plt.imshow(img, cmap="gray")
                col2 = fig.add_subplot(rows, columns, i + 2)
                plt.imshow(mask[j], cmap="gray")
                col3 = fig.add_subplot(rows, columns, i + 3)
                plt.imshow(bin_genmask[j], cmap="gray")
                j += 1
                if i == 0:
                    col1.title.set_text("Data")
                    col2.title.set_text("Ground Truth")
                    col3.title.set_text("Prediction")
            k = k+1
            plt.suptitle(f"Predictions", fontsize=16)
            plt_out_pth = os.path.join("/home/ramat/experiments/exp_tinyCD/exp136/prediction_image", f"prediction_image_{k}.png")
            plt.savefig(plt_out_pth, dpi=300)


def run():

    # set the random seed
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # number of epochs 
    epochs = 100

    # Parse arguments:
    args = parse_arguments()

    # Initialize tensorboard:
    writer = SummaryWriter(log_dir=args.log_path)

    # Inizialitazion of dataset and dataloader:
    trainingdata = MyDataset(args.datapath, "data/train_totalSegmentor.txt", "train")
    validationdata = MyDataset(args.datapath, "data/val_totalSegmentor.txt", "val")
    testingdata = MyDataset("/home/ramat/data/images/test_data_binary", "data/test.txt", "val")
    data_loader_training = DataLoader(trainingdata, batch_size=8, shuffle=True)
    data_loader_val = DataLoader(validationdata, batch_size=8, shuffle=True)
    data_loader_testing = DataLoader(testingdata, batch_size=3, shuffle=False)


    # device setting for training
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')

    print(f'Current Device: {device}\n')

    # Initialize the model
    model = Model()
    restart_from_checkpoint = False
    model_path = None
    if restart_from_checkpoint:
        model.load_state_dict(torch.load(model_path))
        print("Checkpoint succesfully loaded")

    # print number of parameters
    parameters_tot = 0
    for nom, param in model.named_parameters():
        # print (nom, param.data.shape)
        parameters_tot += torch.prod(torch.tensor(param.data.shape))
    print("Number of model parameters {}\n".format(parameters_tot))

    # define the loss function for the model training.
    criterion = torch.nn.BCELoss()

    # choose the optimizer in view of the used dataset
    # Optimizer with tuned parameters for LEVIR-CD
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00356799066427741,
                                  weight_decay=0.009449677083344786, amsgrad=False)

    # Optimizer with tuned parameters for WHU-CD
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.002596776436816101,
    #                                 weight_decay=0.008620171028843307, amsgrad=False)

    # scheduler for the lr of the optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100)

    # copy the configurations
    _ = shutil.copytree(
        "./models",
        os.path.join(args.log_path, "models"),
    )

    train(
        data_loader_training,
        data_loader_val,
        model,
        criterion,
        optimizer,
        scheduler,
        args.log_path,
        writer,
        epochs=epochs,
        save_after=1,
        device=device
    )
    writer.close()
    model = Model()
    model.load_state_dict(torch.load(os.path.join(args.log_path, "model_{}.pth".format(epochs - 1))))
    #output_test_image(data_loader_testing, model, device, os.path.join(args.log_path, "images"))


if __name__ == "__main__":
    run()
