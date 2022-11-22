import os
import cv2
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
    for file in tqdm.tqdm(os.listdir(src_dir)):
        if file[-3:] == "zip":
            with zipfile.ZipFile(os.path.join(src_dir, file), "r") as zip_ref:
                zip_ref.extractall(dst_dir)

def move_files(code_names, channel, img_dir, org_dir):
    for code_name in tqdm.tqdm(code_names):
        cur_dir = code_name+channel
        dst_dir = os.path.join(img_dir, channel)
        for file in os.listdir(os.path.join(org_dir, cur_dir)):
            if file[-3:] == "tif":
                shutil.copy(
                    os.path.join(org_dir, cur_dir, file),
                    os.path.join(dst_dir, file[:12]+file[13:]),
                )


def rearrange_unzipped(dir_path, data_path):
    print("rearranging unzipped files...")
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
                Process(target=move_files, args=(code_dict[channel], channel,  img_dir, org_dir))
            )

        for proc in proc_list:
            proc.start()
        for proc in proc_list:
            proc.join()        
        
        for file in os.listdir(os.path.join(img_dir, "R")):
            shutil.copy(
                os.path.join(
                    png_dir, file[:12] + "R", file[:12] + "R" + file[12:-3] + "png"
                ),
                os.path.join(anno_dir, file[:-3] + "png")
            )
        print("Re-arrangement done")
        print(f"img_dir: {os.path.abspath(img_dir)}, anno_dir: {os.path.abspath(anno_dir)}")

    #Sanity check
    ann_file_list = sorted(os.listdir(anno_dir))
    ann_file_names = []
    for ann_file in ann_file_list:
        ann_file_names.append(ann_file[:-4])

    for channel in tqdm.tqdm(channels):
        img_file_list = sorted(os.listdir(os.path.join(img_dir, channel)))
        img_file_names = []
        for img_file in img_file_list:
            img_file_names.append(img_file[:-4])
        
        assert img_file_names == ann_file_names
        # ic(code_list)
    print("Sanity check is done!")

def ann_rgb2cls(data_path, replace=False):
    print("Converting annotation images to class images...")
    nocls = np.array([0, 0, 0])
    cls1 = np.array([245, 39, 8])   #정상
    cls2 = np.array([245, 299, 0])  #도열병
    cls3 = np.array([26, 0, 255])   #도복
    cls4 = np.array([204, 0, 250])  #결주
    cls5 = np.array([0, 123, 245])  #부진
    class_list = [nocls, cls1, cls2, cls3, cls4, cls5]
    dx = [0, 0, -1, 1, -1, -1, 1, 1]
    dy = [1, -1, 0, 0, -1, 1, -1, 1]

    def remove_blur(mask, cls_img):
        w, h = cls_img.shape[:2]
        for x in range(w):
            for y in range(h):
                if not mask[x, y]:
                    valid = []
                    for d in range(8):
                        nx, ny = x+dx[d], y+dy[d]
                        if 0<= nx< w and 0<= ny <h:
                            valid.append(cls_img[nx][ny])
                    cls_img[x][y] = np.argmax(np.bincount(valid))
        return cls_img

    def mapping_without_blur(file_path):
        img = cv2.imread(file_path)
        img = np.array(img)
        dist = []
        for c in class_list: #각 클래스와의 거리 구하기
            dist.append(np.sum((img - c)*(img - c), axis=2))
        dist = np.array(dist)
        min_dist = np.min(dist, axis= 0)
        mask = np.where(min_dist == 0, True, False)
        mapped_img = np.argmin(dist, axis= 0) #가장 가까운 거리의 인덱스 할당
        result = remove_blur(mask, mapped_img)
        return result
    
    rgb_ann_dir = os.path.join(data_path, "rgb_ann")
    cls_ann_dir = os.path.join(data_path, "cls_ann")

    if not replace and os.path.exists(cls_ann_dir):
        print(f"using existing class annotation files: {os.path.abspath(cls_ann_dir)}")
        return

    os.makedirs(cls_ann_dir, exist_ok=True)
    file_list = os.listdir(rgb_ann_dir)
    try:
        file_list.remove('.ipynb_checkpoints')
    except:
        pass

    for file in tqdm.tqdm(file_list):
        result = mapping_without_blur(os.path.join(rgb_ann_dir, file))
        cv2.imwrite(os.path.join(cls_ann_dir, file), result)
    print("Converting annotation images to class images is done!")


def img_compose(data_path, channels=["E", "N", "G"], mode='chw_minmax', replace=False):
    img_dir = os.path.join(data_path, "img")
    img_conf_name = ''.join(channels)+'_'+mode
    output_dir = os.path.join(data_path, img_conf_name)
    if not replace:
        if os.path.exists(output_dir):
            print(f"using existing img_dir: {os.path.abspath(output_dir)}")
            return img_conf_name
    os.makedirs(output_dir, exist_ok=True)

    if mode == 'chw_minmax':
        # Channel-wise minmax normalization
        for file in tqdm.tqdm(sorted(os.listdir(os.path.join(img_dir, channels[0])))):
            img = np.zeros((512, 512, len(channels)))
            org_chns = []
            for ch_id, channel in enumerate(channels):
                cur_chn = np.array(
                        Image.open(os.path.join(img_dir, channel, file))
                    )
                cur_chn[cur_chn<0] = cur_chn[cur_chn!=-10000].min()
                cur_chn = (cur_chn-cur_chn.min())/cur_chn.ptp()
                img[:, :, ch_id] = cur_chn
            cv2.imwrite(os.path.join(output_dir, file[:-3]+'png'), img*255)
    elif mode == 'minmax':
        # minmax normalization
        for file in tqdm.tqdm(sorted(os.listdir(os.path.join(img_dir, channels[0])))):
            img = np.zeros((512, 512, len(channels)))
            for ch_id, channel in enumerate(channels):
                org_cur_chn = np.array(
                        Image.open(os.path.join(img_dir, channel, file))
                    )
                img[:, :, ch_id] = org_cur_chn
            img[img==-10000] = np.min(img[img!=-10000])
            img = (img-img.min())/img.ptp()
            cv2.imwrite(os.path.join(output_dir, file[:-3]+'png'), img*255)
    elif mode == 'cv2_merge':
        # cv2.merge automatically normalize
        for file in tqdm.tqdm(sorted(os.listdir(os.path.join(img_dir, channels[0])))):
            img = np.zeros((512, 512, len(channels)))
            org_chns = []
            for ch_id, channel in enumerate(channels):
                org_cur_chn = cv2.imread(os.path.join(img_dir, channel, file), cv2.IMREAD_UNCHANGED)
                org_chns.append(org_cur_chn)
            img = cv2.merge(org_chns)
            cv2.imwrite(os.path.join(output_dir, file[:-3]+'png'), img*255)
    else:
        raise NotImplementedError

    return img_conf_name
        

def ann_split(ann_dir, output_dir, split_ratio=[0.8, 0.1, 0.1], replace=False):
    if not replace:
        if os.path.exists(output_dir):
            print(f"using existing splited ann_dir: {os.path.abspath(output_dir)}")
            return
    os.makedirs(output_dir, exist_ok=True)
    ann_list = os.listdir(ann_dir)
    try:
        ann_list.remove('.ipynb_checkpoints')
    except:
        pass

    random.shuffle(ann_list)
    split_names = ['train', 'val', 'test']
    start = 0
    for i, ratio in enumerate(split_ratio):
        end = start + int(len(ann_list)*ratio)
        split_list = ann_list[start:end]
        with open(os.path.join(output_dir, '..', split_names[i]+'.txt'), 'w') as f:
            f.write('\n'.join(split_list))
        split_dir = os.path.join(output_dir, split_names[i])
        os.makedirs(split_dir, exist_ok=True)
        for file in split_list:
            shutil.copy(os.path.join(ann_dir, file), os.path.join(split_dir, file))
        start = end
    print("Splitting annotation images is done!")

def img_split(data_dir, split_dir, img_conf_name, replace=False):
    img_dir = os.path.join(data_dir, img_conf_name)
    output_dir = os.path.join(split_dir, img_conf_name)
    split_names = ['train', 'val', 'test']

    if not replace:
        if os.path.exists(output_dir):
            print(f"using existing img_dir: {os.path.abspath(output_dir)}")
            return
    os.makedirs(output_dir, exist_ok=True)

    for split in split_names:
        split_img_dir = os.path.join(output_dir, split)
        os.makedirs(split_img_dir, exist_ok=True)
        with open(os.path.join(split_dir, split+'.txt'), 'r') as f:
            split_list = f.read().splitlines()
        for file in split_list:
            shutil.copy(os.path.join(img_dir, file), os.path.join(split_img_dir, file))

# TODO: rgb ann to class_id
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Rice data processing", description="What the program does"
    )
    parser.add_argument("--src", type=str, default="../rice_raw_data")
    parser.add_argument("--dst", type=str, default="../rice_unzipped")
    parser.add_argument("--skip_unzip", action="store_true")
    args = parser.parse_args()


    map(lambda x: shutil.rmtree(x), glob.glob(os.path.join('../data', ".ipynb_checkpoints"), recursive=True))
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

    # rearrange_unzipped
    rearrange_unzipped(args.dst, '../data')

    # remove .ipynb_checkpoints

    # compose images
    img_conf_name = img_compose('../data', channels=["G", "N", "E"])
    ann_rgb2cls('../data', replace=False)

    # split train/val/test and save
    ann_split('../data/cls_ann', '../split/ann', replace=True)
    img_split('../data', '../split', img_conf_name, replace=True)
