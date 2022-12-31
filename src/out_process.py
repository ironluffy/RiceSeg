import os
import csv
import datetime
import tqdm
import torch
import pickle
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="output file process", description="What the program does"
    )
    parser.add_argument("--img_config", type=str, default="GNE_chw_minmax")
    parser.add_argument("--data_path", type=str, default="../split")
    parser.add_argument("--src", type=str, default="./out.pkl")
    parser.add_argument("--dst", type=str, default="../out.csv")
    parser.add_argument("--skip_unzip", action="store_true")
    args = parser.parse_args()

    with open(args.src, "rb") as f:
        data = pickle.load(f)

    img_files = os.listdir(os.path.join(args.data_path, args.img_config, "test"))
    ann_files = os.listdir(os.path.join(args.data_path, "ann", "test"))
    assert len(img_files) == len(ann_files)
    assert len(img_files) == len(data)

    for i in range(len(img_files)):
        img_files[i] = img_files[i][:-4]
        ann_files[i] = ann_files[i][:-4]
        assert img_files[i] == ann_files[i]
    print("Sanity check done")

    with open(args.dst, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label", "mAcc", "Time stamp"])
        for i in tqdm.trange(len(img_files)):
            valid_ind = data[i][3] != 0
            mid_dat = data[i][0][valid_ind] / data[i][3][valid_ind] * 100
            if torch.sum(valid_ind) == 0:
                mAcc = "N/A"
            else:
                mAcc = torch.mean(mid_dat[1:]).item()
            writer.writerow(
                [
                    img_files[i],
                    ann_files[i],
                    mAcc,
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ]
            )
