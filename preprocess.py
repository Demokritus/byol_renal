#!/usr/bin/env python3
import os
# import torch
import numpy as np
from typing import List, Tuple
from roi_tensor_extract import ROIExtract
from lightly.data import LightlyDataset
from PIL import Image
import torchvision.transforms as transforms
from SpotGaussianBlur import SpotGaussianBlur
from shuffle_crops import getIsles, drawRectangles, checkOverlap


# the path to the dataset
path_to_data = "/home/gsergei/data/binary_dataset_mix"
# the creation of the dataset for training
dataset = LightlyDataset(path_to_data + "/train")
path_rois_train = path_to_data + "/rois_train"
path_blur_train = path_to_data + "/blur_train"
path_crop_shuffle_train = path_to_data + "/crop_shuffle_train"

# if not os.path.exists(path_rois_train):
    # os.mkdir(path_rois_train)

# if not os.path.isdir(path_blur_train):
#     os.mkdir(path_blur_train)

if not os.path.isdir(path_crop_shuffle_train):
    os.mkdir(path_crop_shuffle_train)


def dataset2rois(dataset: LightlyDataset, path_rois: str) -> None:
    dataset_ = dataset.dataset
    for i in range(len(dataset_)):
        fname = dataset.index_to_filename(dataset_, i)
        sample, _ = dataset_[i]
        # convert PIL Image sample to BGR format
        # sample = sample.convert("1")
        sampleT = transforms.ToTensor()(sample) #.unsqueeze(0)
        print("Sample T shape: ", sampleT.shape)
        roi = ROIExtract()(sampleT)
        print("Data type of roi %s" % type(roi))
        
        if roi is not None:
            # roi.save(path_rois + "/" + fname)
            np_roi = np.array(roi).astype(np.uint8)
            print("Data type of np_roi %s" % type(np_roi))
            # print(np_roi.dtype)
            # print(np_roi.shape)
            np_roi = np_roi.reshape((np_roi.shape[1], np_roi.shape[2]))
            img = Image.fromarray(np_roi)
            img.save(path_rois + "/" + fname)
        else:
            print("No ROI found for " + fname)


def dataset2blurr(dataset: LightlyDataset, path_blur: str) -> None:
    dataset_ = dataset.dataset
    for i in range(len(dataset_)):
        fname = dataset.index_to_filename(dataset_, i)
        sample, _ = dataset_[i]
        # convert PIL Image sample to BGR format
        # sample = sample.convert("1")
        # Using PIL convert 3-channel sample to 1-channel
        sample = sample.convert("L")
        sampleT = transforms.ToTensor()(sample) # .unsqueeze(0)
        # transpose sampleT to (H, W, C)
        # sampleT = sampleT.permute(2, 3, 1, 0)
        print("Sample T shape: ", sampleT.shape)
        # roi = ROIExtract()(sampleT)
        new_sample = SpotGaussianBlur(sampleT)()
        # print("Data type of roi %s" % type(new_sample))
        
        if new_sample is not None:
            # roi.save(path_rois + "/" + fname)
            np_roi = np.array(new_sample).astype(np.uint8)
            print("Data type of np_roi %s" % type(np_roi))
            print("NP ROI SHAPE is {}".format(np_roi.shape))
            # print(np_roi.dtype)
            # print(np_roi.shape)
            np_roi = np_roi.reshape((np_roi.shape[1], np_roi.shape[2]))
            np_roi = (np_roi * 255).astype(np.uint8)
            img = Image.fromarray(np_roi)
            img.save(path_blur + "/" + fname)
        else:
            print("No ROI found for " + fname)


def dataset2cropshuffle(dataset: LightlyDataset, path: str) -> None:
    dataset_ = dataset.dataset
    for i in range(len(dataset_)):
        fname = dataset.index_to_filename(dataset_, i)
        sample, _ = dataset_[i]
        # convert PIL Image sample to BGR format
        # sample = sample.convert("1")
        # Using PIL convert 3-channel sample to 1-channel
        sample = sample.convert("L")
        sampleT = transforms.ToTensor()(sample) # .unsqueeze(0)
        sampleT = sampleT.permute(1,2,0)
        # sampleT = np.array(sampleT).astype(np.uint8)
        positions, segments = getIsles(sampleT)
        new_sample = drawRectangles(sampleT, positions)
        
        if new_sample is not None:
            # roi.save(path_rois + "/" + fname)
            np_roi = np.array(new_sample).astype(np.uint8)
            print("Data type of np_roi %s" % type(np_roi))
            print("NP ROI SHAPE is {}".format(np_roi.shape))
            # print(np_roi.dtype)
            # print(np_roi.shape)
            np_roi = np_roi.reshape((np_roi.shape[0], np_roi.shape[1]))
            np_roi = (np_roi * 255).astype(np.uint8)
            img = Image.fromarray(np_roi)
            
            if np.all(np.unique(new_sample) == 0):
                # img.save(path + "/" + fname)
                print("EMPTY TENSOR! {}".format(fname))
            else:
                print("GOOD TENSOR {}".format(fname))
            
            img.save(path + "/" + fname)
        else:
            # print("No ROI found for " + fname)
            print("None type for " + fname + " returned")


if __name__ == "__main__":
    # dataset2rois(dataset, path_rois_train)
    # dataset2blurr(dataset, path_blur_train)    
    dataset2cropshuffle(dataset, path_crop_shuffle_train)
        



