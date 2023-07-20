import torch
from dataset.dataset import MyDataset
import tqdm
from torch.utils.data import DataLoader
from metrics.metric_tool import ConfuseMatrixMeter
from models.change_classifier import ChangeClassifier
import argparse
from os.path import join
import matplotlib.pyplot as plt
import numpy as np

MEAN = 0.45
STD = 0.225

def denormalize_img(img:np.array)->np.array:
    return (img * STD) + MEAN

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
        "--modelpath",
        type=str,
        help="model path",
    )
    parser.add_argument(
        "--save_pred_path",
        type=str,
        help="save predictions path",
    )

    parsed_arguments = parser.parse_args()
    
    return parsed_arguments


def compose_mask(
        image: np.ndarray,
        mask: np.ndarray,
        channel: int = 0,
        alpha: float = 0.8,
    ) -> np.ndarray:
    if len(image.shape) < 3:
        image = image[:,:,None]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    res = image.copy()
    res *= alpha
    res[:,:,channel] += mask * (1 - alpha)
    return res


def main():
    # Parse arguments:
    args = parse_arguments()

    # tool for metrics
    tool_metric = ConfuseMatrixMeter(n_class=2)

    # Initialisation of the dataset
    data_path = args.datapath 
    dataset = MyDataset(data_path, "data/test_totalSegmentor.txt", "test")
    test_loader = DataLoader(dataset, batch_size=1)

    # Initialisation of the model and print model stat
    model = ChangeClassifier()
    modelpath = args.modelpath
    model.load_state_dict(torch.load(modelpath))

    # Print the number of model parameters 
    param_tot = sum(p.numel() for p in model.parameters())
    print()
    print("Number of model parameters {}".format(param_tot))
    print()

    # Set evaluation mode and cast the model to the desidered device
    model.eval()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model.to(device)

    # loop to evaluate the model and print the metrics
    bce_loss = 0.0
    criterion = torch.nn.BCELoss()

    with torch.no_grad():
        for data in tqdm.tqdm(test_loader):
            reference = data["image"]
            mask = data["mask"]
            img_name = data["img_name"]
            reference = reference.to(device).float()
            mask = mask.float()

            # pass refence and test in the model
            generated_mask = model(reference).squeeze(1)
            
            # compute the loss for the batch and backpropagate
            generated_mask = generated_mask.to("cpu")
            bce_loss += criterion(generated_mask, mask)

            ### Update the metric tool
            bin_genmask = (generated_mask > 0.5).numpy()
            bin_genmask = bin_genmask.astype(int)
            mask = mask.numpy()
            mask = mask.astype(int)
            tool_metric.update_cm(pr=bin_genmask, gt=mask)

            # Preparing the masks:
            ct_to_plot = denormalize_img(reference[0].detach().cpu().permute(1,2,0).numpy())
            gt_mask = compose_mask(ct_to_plot, mask[0], 2)
            generated_mask = compose_mask(ct_to_plot, bin_genmask[0])
            fp = np.maximum(bin_genmask[0] - mask[0], 0)
            fn = np.maximum(mask[0] - bin_genmask[0], 0)
            diff_mask = compose_mask(ct_to_plot, fp, 0, alpha=0.75)
            diff_mask = compose_mask(diff_mask, fn, 1, alpha=0.8)
            
            # save prediction to folder
            # TODO: check 
            fig, axs = plt.subplots(2,2)
            axs[0,0].imshow(ct_to_plot)
            axs[0,0].set_title("CT scan")
            axs[0,1].imshow(gt_mask,cmap="gray")
            axs[0,1].set_title("GT mask")
            axs[1,0].imshow(generated_mask,cmap="gray")
            axs[1,0].set_title("Predicted mask")
            axs[1,1].imshow(diff_mask,cmap="gray")
            axs[1,1].set_title("Difference mask")
            #

            plt.savefig(fname=join(args.save_pred_path,img_name[0]))
            

        bce_loss /= len(test_loader)
        print("Test summary")
        print("Loss is {}".format(bce_loss))
        print()

        scores_dictionary = tool_metric.get_scores()
        print(scores_dictionary)


if __name__ == "__main__":
    main()
