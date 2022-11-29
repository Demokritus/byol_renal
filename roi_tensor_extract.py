import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import cv2
from typing import *

# import lightly
# global gen_cropped_image ;
# global gen_crop_isles ;
# global original_size ;

original_size : Tuple[int,int] = (1024, 1024)
MIN_SHARE : float = 0.1
NEW_SIZE : Tuple[int,int] = (820, 820)


# class ROIExtract(object):
class ROIExtract(transforms.RandomCrop):
    def __init__(self, size = NEW_SIZE, from_tensor : bool = False,  **kwargs):
        super().__init__(size, **kwargs)
        self.IMG_SIZE = size
        self.from_tensor : bool = from_tensor
    

    def __call__(self, x):
        crop = self.gen_cropped_image(x)
        return crop


    def forward(self, x):
        crop = self.gen_cropped_image(x)
        return crop

    def getROIPointsV2(self, img : torch.Tensor, 
                       min_share : float = MIN_SHARE) -> Tuple[List,List]:
        '''
        img_path : str
            a path to an image,
        min_share : float
            a minimal share of non-black pixels in a given segment (filtering)
            
        ______
        Return:
            a tuple of tuples, first tuple - X and Y of top left corner, 
            second tuple - X and Y of bottom right corner
        '''
        # img = cv2.imread(img_path)
        img = np.array(img).astype(np.uint8)
        # img = np.array(img * 255).astype(np.uint8)

        if img.shape[-3] > 1 and len(img.shape) > 2:
            # print("INSIDE getROIPointsV2 \n \
            #    the shape of IMG is {}".format(img.shape))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cp_img = np.copy(img)
        img_mean = np.mean(img)
        
        img[img < img_mean] = 0
        img[img >= img_mean] = 255
        
        _, thresh = cv2.threshold(cp_img, img_mean, 255, 0)
        contours, _ = cv2.findContours(thresh, 2, 1)

        cv2.fillPoly(thresh, pts = contours, color=(255,0,255))

        #===DILATION===
        kernel = np.ones((3,3), np.uint8)   # set kernel as 3x3 matrix from numpy
        # Create dilation image from the original image
        img_alt = cv2.dilate(thresh, kernel, iterations=15)

        hull = []

        for i in range(len(contours)):
            # the result of convexHull is a list of points
            new_hull = cv2.convexHull(contours[i], returnPoints=True) 
            hull.append()
        
        # ===DRAWING===
        # create an empty black image
        islands = np.zeros((len(contours), thresh.shape[0], thresh.shape[1]))
        # segments = np.zeros((len(contours), thresh.shape[0], thresh.shape[1]))
        segments = []
        for i in range(len(contours)):
            color = 255
            # draw contours for ith island, where hull stores contour points, color - the color of the filling
            # and the last two parameters are thickness and line type
            cv2.drawContours(islands[i, :, :], hull, i, color, -1, 8)
            share = lambda piece: np.sum(piece == 255) / np.size(piece)
            # share = np.sum(islands[i,:,:] == color) / np.size(islands[i,:,:])
            if share(islands[i, :, :]) >= min_share:
                segments.append(islands[i, :, :])
        
        positions = []
        for k in range(len(segments)):
            x1 = np.min(np.where(segments[k])[0])
            y1 = np.min(np.where(segments[k])[1])
            x2 = np.max(np.where(segments[k])[0])
            y2 = np.max(np.where(segments[k])[1])
            positions.append(((x1,y1), (x2,y2)))
        if len(segments) == 0:
            # print("EMPTY SEGMENTS LIST")
            positions.append(((0,0), original_size))
        return positions, segments
    

    def gen_crop_isles(self, img, box_size, positions,
                    isles, min_share,
                    check=None, share=None):
        
        # opening an image using img_path argument
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_mean = np.mean(img)
        else:
            img_mean = 0

        # function returns nothing if the whole image does not meet the criterion
        if not check(img, img_mean, box_size, min_share):
            print("EMPTY IMAGE")
            return None
            # return torch.zeros(1, box_size, box_size)
        

        def pick_k_xy(positions, L=len(positions), c=0):
                # L = len(positions)
                k = np.random.randint(0,L)
                # generating random coordinates of top left corner of a box 
                # inside of an area around tissue blob with coord-s ((x1,y1), (x2,y2))
                pt1, pt2 = positions[k]
                x1, y1 = pt1
                x2, y2 = pt2
                if (x2 - box_size) > x1 and (y2 - box_size) > y1 and c<=3:
                    x = np.random.randint(x1, x2 - box_size)
                    y = np.random.randint(y1, y2 - box_size)
                    return x, y, k
                elif c<=3:
                    c += 1
                    return pick_k_xy(positions, c=c)
                else:
                    return 0,0,0
        
        x, y, k = pick_k_xy(positions)
        (x1,y1), (x2,y2) = positions[k]
        
        if share(isles[k], x1, y1, x2, y2) >= min_share:
            res_tensor : torch.Tensor
            # res_tensor = torch.Tensor(img[x:x+box_size, y:y+box_size, 0])
            # res_tensor = torch.transpose(img, 0, 2)
            img = torch.Tensor(img)
            res_tensor = img.permute(2, 0, 1)
            return res_tensor[:1, x:x+box_size, y:y+box_size]

        else:
            # gen_crop_isles(img, min_share=min_share, box_size=box_size)
            self.gen_crop_isles(img, box_size, 
                            positions, isles, min_share, 
                                check=check,
                                share=share)
        

    def gen_cropped_image(self, img : torch.Tensor,
                          min_share : float = MIN_SHARE) -> Tuple[np.array, Tuple]:
        '''
        The function generates a randomly cropped image from a randomly chosen ROI on original image
        
        Arguments:
        img_path : str
            a path to an image file,
        min_share : float
            minimal share of tissue island on a given cropped image (threshold),
        box_size : int
            the size of a cropped image
        
        Return:
        cropped_image : torch.Tensor
            a cropped image
        '''
        
        # box size
        box_size : int = self.IMG_SIZE[0]
        # print("INSIDE GEN CROPPED FUN")
        # print("Inpurt tensor values {}".format(torch.unique(img)))
        if not self.from_tensor:
            print("Multiplying by 255")
            img = np.array(img * 255).astype(np.uint8)
        else:
            img = np.array(img).astype(np.uint8)

        if len(img.shape) == 3:
            img = np.transpose(img, (1,2,0))
        elif len(img.shape) == 4:
            img = np.transpose(img, (0,2,3,1))

        # checking a randomly assigned box on an image for number of white pixels
        share = lambda img, x1, y1, x2, y2: \
                        np.sum(img[x1:x2,y1:y2] == 255) / np.size(img[x1:x2,y1:y2])
        
        # checking a whole image for the size of foreground
        check = lambda img, img_mean, box_size, min_share: \
                        np.sum(img >= img_mean) / box_size ** 2 >= min_share
        # check = lambda img, img_mean, box_size, min_share: np.sum(img >= 1.5 * img_mean) / box_size ** 2 >= min_share
        
        # acquiring positions of isles and isles themselves
        positions, isles = self.getROIPointsV2(img, min_share = min_share)

        if len(isles) > 0:
            # print("GEN CROP ISLES call")
            return self.gen_crop_isles(img, box_size, 
                            positions, isles, min_share, 
                                check=check,
                                share=share)
        else:
            print("CONTINUE GEN CROPPED IMAGE call")
            # opening an image using img_path argument
            # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_mean = np.mean(img)
            # function returns nothing if the whole image does not meet the criterion
            if not check(img, img_mean, box_size, min_share):
                print("EMPTY IMAGE")
                # return empty tensor filled with zeroes
                return torch.zeros(1, box_size, box_size)
            
            # picking a random position from a number of positions
            L = len(positions)
            # if L > 0:
            k = np.random.randint(0, L)
            # else:
            #    k = 0

            def pick_k_xy(positions, c=0):
                L = len(positions)
                k = np.random.randint(0,L)
                # generating random coordinates of top left corner of a box 
                # inside of an area around tissue blob with coord-s ((x1,y1), (x2,y2))
                pt1, pt2 = positions[k]
                x1, y1 = pt1
                x2, y2 = pt2
                if (x2 - box_size) > x1 and (y2 - box_size) > y1 and c<=3:
                    x = np.random.randint(x1, x2 - box_size)
                    y = np.random.randint(y1, y2 - box_size)
                    return x, y
                elif c<=3:
                    c += 1
                    return pick_k_xy(positions, c=c)
                else:
                    return 0,0
            
            x, y = pick_k_xy(positions)
            res_tensor : torch.Tensor
            img = torch.Tensor(img)
            res_tensor = img.permute(2, 0, 1)
            return res_tensor[:1, x:x+box_size, y:y+box_size]
