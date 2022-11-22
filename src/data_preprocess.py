import os
import cv2
import tqdm
import shutil
import zipfile
import argparse
import numpy as np
from PIL import Image
from icecream import ic


def unzip_files(src_dir, dst_dir):
    for file in tqdm.tqdm(os.listdir(src_dir)):
        if file[-3:] == "zip":
            with zipfile.ZipFile(os.path.join(src_dir, file), "r") as zip_ref:
                zip_ref.extractall(dst_dir)


def rearrange_unzipped(dir_path, data_path):
    channels = ["R", "G", "B", "N", "E"]
    org_dir = os.path.join(dir_path, "org")
    png_dir = os.path.join(dir_path, "png")

    img_dir = os.path.join(data_path, "img")
    anno_dir = os.path.join(data_path, "rgb_ann")
    

    if os.path.exists(img_dir) and os.path.exists(anno_dir):
        print(
            f"using existing rearranged files\nimg_dir: {os.path.abspath(img_dir)}, anno_dir: {os.path.abspath(anno_dir)}"
        )
    else:
        for channel in channels:
            os.makedirs(os.path.join(img_dir, channel), exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)

        for cur_dir in tqdm.tqdm(os.listdir(org_dir)):
            code_name, channel = cur_dir[:-1], cur_dir[-1:]
            dst_dir = os.path.join(img_dir, channel)
            for file in tqdm.tqdm(os.listdir(os.path.join(org_dir, cur_dir))):
                if file[-3:] == "tif":
                    shutil.copy(
                        os.path.join(org_dir, cur_dir, file),
                        os.path.join(dst_dir, file[:12]+file[13:]),
                    )

        for file in os.listdir(os.path.join(img_dir, "R")):
            shutil.copy(
                os.path.join(
                    png_dir, file[:12] + "R", file[:12] + "R" + file[:-3] + "png"
                ),
                os.path.join(anno_dir, file[:-3] + "png"),
            )
        print("Re-arrangement done")
        print(f"img_dir: {os.path.abspath(img_dir)}, anno_dir: {os.path.abspath(anno_dir)}")

    # Sanity check
    # for code_dir in os.listdir(img_dir):
    #     for channel in channels:
    #         img_file_list = []
    #         for file in os.listdir(os.path.join(img_dir, code_dir, channel)):
    #             img_file_list.append(file[:-3])
    #         ann_file_list = []
    #         for file in os.listdir(os.path.join(anno_dir, code_dir)):
    #             ann_file_list.append(file[:-3])
    #         assert sorted(img_file_list) == sorted(ann_file_list)
    #     # ic(code_list)
    print("Sanity check is done!")

# TODO: img_compose re-write
# def img_compose(data_path, channels=["G", "N", "E"]):
#     img_dir = os.path.join(data_path, "img")
#     output_dir = os.path.join(data_path, ''.join(sorted(channels)))
#     os.makedirs(output_dir, exist_ok=True)
#     img = []
#     for channel in channels:
#         img.append(
#             cv2.imread(
#                 os.path.join(img_dir, code_dir, channel, file), cv2.IMREAD_UNCHANGED
#             )
#         )
#     return np.concatenate(img, axis=2)

# TODO: rgb ann to class_id
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Rice data processing", description="What the program does"
    )
    parser.add_argument("--src", type=str, default="./rice_raw_data")
    parser.add_argument("--dst", type=str, default="./rice_unzipped")
    args = parser.parse_args()

    # unzip
    if not os.path.exists(os.path.join(args.dst)):
        os.makedirs(args.dst)
        unzip_files(src_dir=args.src, dst_dir=args.dst)
        # UTF-8 encoding problem: Korean letters is not properly encoded
        os.rename(
            os.path.join(args.dst, sorted(os.listdir(args.dst))[1]),
            os.path.join(args.dst, "org"),
        )
        print(f"unzip done: {args.dst}")
    else:
        print(f"using existing unzipped files: {os.path.abspath(args.dst)}")

    # rearrange_unzipped
    rearrange_unzipped(args.dst, './data')
