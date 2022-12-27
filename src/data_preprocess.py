import os
import cv2
import json
import glob
import tqdm
import random
import shutil
import zipfile
import argparse
import numpy as np
from PIL import Image
from icecream import ic
from multiprocessing import Process


def unzip_files(src_dir, dst_dir):
    def unzip_file(file):
        try:
            with zipfile.ZipFile(file, "r") as zip_ref:
                zip_ref.extractall(dst_dir)
        except:
            print(f"Error in {file}")

    proc_list = []
    for file in os.listdir(src_dir):
        if file[-3:] == "zip":
            proc_list.append(
                Process(target=unzip_file, args=(os.path.join(src_dir, file),))
            )
    for proc in proc_list:
        proc.start()
    for proc in proc_list:
        proc.join()


def move_files(code_names, channel, img_dir, org_dir):
    for code_name in tqdm.tqdm(code_names):
        cur_dir = code_name + channel
        dst_dir = os.path.join(img_dir, channel)
        for file in os.listdir(os.path.join(org_dir, cur_dir)):
            if file[-3:] == "tif":
                shutil.copy(
                    os.path.join(org_dir, cur_dir, file),
                    os.path.join(dst_dir, file[:12] + file[13:]),
                )


def rearrange_unzipped(dir_path, data_path, replace=False):
    print("rearranging unzipped files...")
    channels = ["R", "G", "B", "N", "E"]
    org_dir = os.path.join(dir_path, "org")


    img_dir = os.path.join(data_path, "img")
    anno_dir = os.path.join(data_path, "cls_ann")

    if os.path.exists(img_dir) and os.path.exists(anno_dir) and not replace:
        print(
            f"using existing rearranged files\nimg_dir: {os.path.abspath(img_dir)}, anno_dir: {os.path.abspath(anno_dir)}"
        )
    else:
        for channel in channels:
            os.makedirs(os.path.join(img_dir, channel), exist_ok=True)
        os.makedirs(anno_dir, exist_ok=True)

        code_dict = {}
        for channel in channels:
            code_dict[channel] = []
        for cur_dir in os.listdir(org_dir):
            code_name, channel = cur_dir[:-1], cur_dir[-1:]
            code_dict[channel].append(code_name)

        for channel in channels:
            assert sorted(code_dict[channel]) == sorted(code_dict["R"])

        proc_list = []
        for channel in channels:
            proc_list.append(
                Process(
                    target=move_files,
                    args=(code_dict[channel], channel, img_dir, org_dir),
                )
            )

        for proc in proc_list:
            proc.start()
        for proc in proc_list:
            proc.join()

        # for file in os.listdir(os.path.join(img_dir, "R")):
        #     shutil.copy(
        #         os.path.join(
        #             png_dir, file[:12] + "R", file[:12] + "R" + file[12:-3] + "png"
        #         ),
        #         os.path.join(anno_dir, file[:-3] + "png"),
        #     )
        print("Re-arrangement done")
        print(
            f"img_dir: {os.path.abspath(img_dir)}, anno_dir: {os.path.abspath(anno_dir)}"
        )

    # Sanity check
    print("Sanity check... (img_dir, anno_dir)")
    ann_file_list = sorted(os.listdir(anno_dir))
    ann_file_names = []
    for ann_file in ann_file_list:
        ann_file_names.append(ann_file[:-4])

    for channel in tqdm.tqdm(channels):
        img_file_list = sorted(os.listdir(os.path.join(img_dir, channel)))
        img_file_names = []
        for img_file in img_file_list:
            if img_file[:-4] not in ann_file_names:
                os.remove(os.path.join(img_dir, channel, img_file))
                continue
            img_file_names.append(img_file[:-4])

        assert ann_file_names == img_file_names
        # ic(code_list)
    print("Sanity check is done!")



def img_compose(
    data_path: str, channels=["E", "N", "G"], mode="chw_minmax", replace=False
) -> str:
    """Compose png images from channel-wise data (mostly in .tif format)

    Args:
        data_path (str): the path of the 'data' directory
        channels (list, optional): Among ["R", "G", "B", "N", "E], select three (repetition allowed). Defaults to ["E", "N", "G"].
        mode (str, optional): float32 to int8 convert and normalize method. Defaults to 'chw_minmax'.
        replace (bool, optional): replace existing results or not. Defaults to False.

    Raises:
        NotImplementedError: when the mode name is not supported

    Returns:
        str: img_conf_name
    """
    print("Composing images...")
    assert len(channels) == 3
    img_dir = os.path.join(data_path, "img")
    img_conf_name = "".join(channels) + "_" + mode
    output_dir = os.path.join(data_path, img_conf_name)
    if not replace:
        if os.path.exists(output_dir):
            print(
                f"For image composition, using existing img_dir: {os.path.abspath(output_dir)}"
            )
            return img_conf_name
    os.makedirs(output_dir, exist_ok=True)

    if mode == "chw_minmax":
        # Channel-wise minmax normalization
        for file in tqdm.tqdm(sorted(os.listdir(os.path.join(img_dir, channels[0])))):
            img = []
            for ch_id, channel in enumerate(channels):
                cur_chn = np.array(Image.open(os.path.join(img_dir, channel, file)))
                cur_chn[cur_chn < 0] = cur_chn[cur_chn != -10000].min()
                cur_chn = (cur_chn - cur_chn.min()) / cur_chn.ptp()
                img.append(cur_chn)
            img = np.stack(img, axis=2)
            cv2.imwrite(os.path.join(output_dir, file[:-3] + "png"), img * 255)
    elif mode == "minmax":
        # minmax normalization
        for file in tqdm.tqdm(sorted(os.listdir(os.path.join(img_dir, channels[0])))):
            img = []
            for ch_id, channel in enumerate(channels):
                cur_chn = np.array(Image.open(os.path.join(img_dir, channel, file)))
                img.append(cur_chn)
            img = np.stack(img, axis=2)
            img[img == -10000] = np.min(img[img != -10000])
            img = (img - img.min()) / img.ptp()
            cv2.imwrite(os.path.join(output_dir, file[:-3] + "png"), img * 255)
    elif mode == "cv2_merge":
        # cv2.merge automatically normalize
        for file in tqdm.tqdm(sorted(os.listdir(os.path.join(img_dir, channels[0])))):
            org_chns = []
            for ch_id, channel in enumerate(channels):
                org_cur_chn = cv2.imread(
                    os.path.join(img_dir, channel, file), cv2.IMREAD_UNCHANGED
                )
                org_chns.append(org_cur_chn)
            img = cv2.merge(org_chns)
            cv2.imwrite(os.path.join(output_dir, file[:-3] + "png"), img * 255)
    else:
        raise NotImplementedError

    print("Image composition is done!")
    return img_conf_name


def ann_split(ann_dir, output_dir, split_ratio=[0.8, 0.1, 0.1], replace=False):
    if not replace:
        if os.path.exists(output_dir):
            print(f"using existing splited ann_dir: {os.path.abspath(output_dir)}")
            return
    os.makedirs(output_dir, exist_ok=True)
    ann_list = os.listdir(ann_dir)
    try:
        ann_list.remove(".ipynb_checkpoints")
    except:
        pass

    random.shuffle(ann_list)
    split_names = ["train", "val", "test"]
    start = 0
    for i, ratio in enumerate(split_ratio):
        end = start + int(len(ann_list) * ratio)
        split_list = ann_list[start:end]
        with open(os.path.join(output_dir, "..", split_names[i] + ".txt"), "w") as f:
            f.write("\n".join(split_list))
        split_dir = os.path.join(output_dir, split_names[i])
        os.makedirs(split_dir, exist_ok=True)
        for file in split_list:
            shutil.copy(os.path.join(ann_dir, file), os.path.join(split_dir, file))
        start = end
    print("Splitting annotation images is done!")


def img_split(data_dir, split_dir, img_conf_name, replace=False):
    img_dir = os.path.join(data_dir, img_conf_name)
    output_dir = os.path.join(split_dir, img_conf_name)
    split_names = ["train", "val", "test"]

    if not replace:
        if os.path.exists(output_dir):
            print(f"using existing img_dir: {os.path.abspath(output_dir)}")
            return
    os.makedirs(output_dir, exist_ok=True)

    for split in split_names:
        split_img_dir = os.path.join(output_dir, split)
        os.makedirs(split_img_dir, exist_ok=True)
        with open(os.path.join(split_dir, split + ".txt"), "r") as f:
            split_list = f.read().splitlines()
        for file in split_list:
            shutil.copy(os.path.join(img_dir, file), os.path.join(split_img_dir, file))


def code2cls(obj_cls_code):
    if obj_cls_code == '01':
        cls = 1
    elif obj_cls_code == '02':
        cls = 2
    elif obj_cls_code == '03':
        cls = 3
    elif obj_cls_code == '04':
        cls = 4
    elif obj_cls_code == '05':
        cls = 5
    else:
        cls = 0
    return cls 

def json2clsmask(json_dir, output_dir, replace=False):
    print("Converting json to clsmask...")
    if not replace:
        if os.path.exists(output_dir):
            print(f"using existing clsmask_dir: {os.path.abspath(output_dir)}")
            return
    os.makedirs(output_dir, exist_ok=True)
    all_files = os.listdir(json_dir)
    R_files = []
    for file in all_files:
        if file[-10]=='R':
            R_files.append(file)

    for file in tqdm.tqdm(R_files):
        with open(os.path.join(json_dir, file), "r") as f:
            data = json.load(f)
        mask = np.zeros(data['IMAGES']['PHOTO_FILE_MG'], dtype=np.uint8)
        objs = sorted(data['ANNOTATIONS'], key=lambda x: code2cls(x['OBJECT_CLASS_CODE']))
        for obj in objs:
            cls = code2cls(obj['OBJECT_CLASS_CODE'])
            x, y = obj['PYN_XCRDNT'], obj['PYN_YCRDNT']
            coords = np.stack([x, y], axis=0).transpose()
            cv2.fillPoly(mask, [coords], cls)
        cv2.imwrite(os.path.join(output_dir, file[:-10]+file[-9:-4] + "png"), mask)
    
    print("Converting json to clsmask is done!")
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Rice data processing", description="What the program does")
    parser.add_argument("--src", type=str, default="../rice_raw_data")
    parser.add_argument("--dst", type=str, default="../rice_unzipped")
    parser.add_argument("--skip_unzip", action="store_true")
    args = parser.parse_args()

    map(
        lambda x: shutil.rmtree(x),
        glob.glob(os.path.join("../data", ".ipynb_checkpoints"), recursive=True),
    )
    # unzip
    if not args.skip_unzip:
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
    else:
        print("skip unzip")

    json2clsmask("../json_labels", "../data/cls_ann", replace=False)

    # rearrange_unzipped
    rearrange_unzipped(args.dst, "../data", replace=False)

    # compose images
    img_conf_name = img_compose("../data", channels=["R", "N", "E"])

    # json file to mask

    # split train/val/test and save
    # split annotation first and then split images along with the annotation
    ann_split("../data/cls_ann", "../split/ann", replace=True)
    img_split("../data", "../split", img_conf_name, replace=True)


