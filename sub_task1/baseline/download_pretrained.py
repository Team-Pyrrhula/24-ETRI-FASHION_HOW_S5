import timm
import torch
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args, _ = parser.parse_known_args()
    
    model = args.model
    model_name = model.split("/")[1]
    model = timm.create_model(model, pretrained=True)
    torch.save(model.state_dict(), f"./save/pretrained/{model_name}.pth")