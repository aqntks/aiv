
import os
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset

from aiv.model import DonutModel
from aiv.evaluator import OCREvaluator
from aiv.utils import visualization


def inference(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)

    pretrained_model = DonutModel.from_pretrained(args.pretrained_model_name)

    if args.fp16:
        pretrained_model.half()
        
    pretrained_model.to(device)
    pretrained_model.eval()

    dataset = load_dataset(args.dataset_path, split=args.split)

    preds, gts = [], []
    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        
        ground_truth = sample["ground_truth"]
        gt = ground_truth["gt_parse"]

        output = pretrained_model.inference(image=sample["image"], prompt=f"<s_{args.task_name}>")["predictions"][0] 

        preds.append(output)
        gts.append(gt)

        if args.save_path:
            vis_img = visualization(sample["image"], output["ocr"], output["box"])
            vis_img.save(os.path.join(args.save_path, f"{idx}.png")) 

    evaluator = OCREvaluator(preds, gts)

    acc = evaluator.calculate_text_accuracy()
    precision, recall = evaluator.calculate_box_precision_and_recall()

    if args.save_path:
        scores = {
            "text_accuracy": acc,
            "box_precision": precision,
            "box_recall": recall
        }
        with open(os.path.join(args.save_path, "scores.json"), "w") as f:
            json.dump(scores, f)
      

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="./output")
    parser.add_argument("--fp16", type=bool, default=True)
    args, left_argv = parser.parse_known_args()

    inference(args)
