import os
import sys
import torch
import numpy as np
from shapely.geometry import Point, Polygon
import cv2
from typing import Tuple, List
import torchvision.transforms as T
import torchvision.transforms as transforms


MIN_SHARE : float = 0.1


def getIsles(img) -> Tuple[List, List]:
        '''
        img_path : torch.Tensor
            an image in the form of a torch.Tensor,
        min_share : float
            a minimal share of non-black pixels in a given segment (filtering)
            
        ______
        Return:
            a tuple of tuples, first tuple - X and Y of top left corner, 
            second tuple - X and Y of bottom right corner
        '''
        img = np.array(img).astype(np.uint8)
            
        cp_img = np.copy(img)
        img_mean = np.mean(img)
        
        _, thresh = cv2.threshold(cp_img, img_mean, 255, 0)

        contours, _ = cv2.findContours(thresh, 2, 1)
        # cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 2, 1)
        
        cv2.fillPoly(thresh, pts = contours, color=(255,0,255))

        #===DILATION===
        kernel = np.ones((3,3), np.uint8)   # set kernel as 3x3 matrix from numpy
        # Create dilation image from the original image
        img_alt = cv2.dilate(thresh, kernel, iterations=15)

        hull = []
        positions = []

        for i in range(len(contours)):
            # the result of convexHull is a list of points
            new_hull = cv2.convexHull(contours[i], returnPoints=True)
            # pdb.set_trace() 
            hull.append(new_hull)
        
        # ===DRAWING===
        # create an empty black image
        print("Threshold shape is {}".format(thresh.shape))
        islands = np.zeros((len(contours), thresh.shape[0], thresh.shape[1]))
        print("Islands shape is {}".format(islands.shape))
        # segments = np.zeros((len(contours), thresh.shape[0], thresh.shape[1]))
        
        
        segments = []
        for i in range(len(contours)):
            color = 255
            # draw contours for ith island, where hull stores contour points, color - the color of the filling
            # and the last two parameters are thickness and line type
            cv2.drawContours(islands[i, :, :], hull, i, color, -1, 8)
            
            segments.append(islands[i, :, :])
        
        print("Islands {}".format(islands.shape))

        # positions = []
        for k in range(len(segments)):
            x1 = np.min(np.where(segments[k])[0])
            y1 = np.min(np.where(segments[k])[1])
            x2 = np.max(np.where(segments[k])[0])
            y2 = np.max(np.where(segments[k])[1])
            positions.append(((x1,y1), (x2,y2)))
        if len(segments) == 0:
            print("EMPTY SEGMENTS LIST")
            positions.append(((0,0), img.shape))
        
        return positions, contours
    
    
# create a function that receive a list of ROI positions and their sizes
# then it creates an empty image of the same size as the original image
# and inserts the ROIs into the empty image using the positions
# the function returns this new image with randomly positioned ROIs that do not overlap
def drawRectangles(img, positions):
    '''
    img : torch.Tensor
        an image in the form of a torch.Tensor
    positions : list
        they store coordinates of the top left and bottom right corners of the ROIs
        a list of tuples, first tuple - X and Y of top left corner, 
        second tuple - X and Y of bottom right corner
    '''
    # create an empty image of the same size as the original image
    newImg = np.zeros(img.shape)
    newPositions = [] # they store coordinates of the top left and bottom right corners of the inserted ROIs
    # draw rectangles on it randomly but without overlapping
    for i in range(len(positions)):
        # pdb.set_trace()
        # cv2.rectangle(img, positions[i][0], positions[i][1], (255, 0, 0), 1)
        # generate random position for the rectangle
        # check if it overlaps with any of the other rectangles that have been drawn already
        # if it does, generate another random position and check again until it does not overlap
        newPos = np.random.randint(0, img.shape[0], 2)
        
        try:
            boxSize = (positions[i][1][0] - positions[i][0][0], \
                positions[i][1][1] - positions[i][0][1])
        except IndexError:
            print("IndexError! Skipping this iteration...")
            continue
        finally:
            pass
        
        newPositions.append((newPos, (newPos[0] + boxSize[0], newPos[1] + boxSize[1])))
        
        # check if it overlaps with any of the other rectangles that have been drawn already
        # if it does, generate another random position and check again until it does not overlap
        c: int = 0
        while checkOverlap(newPos, boxSize, newPositions) and c < 50:
            newPos = np.random.randint(0, img.shape[0], 2)
            c += 1
        
        if checkOverlap(newPos, boxSize, newPositions):
            print("Overlap! Skipping this iteration...")
            continue
        
        # insert the ROI located at positions[i] on old image into the new image at the new position
        print("Inserting ROI at position {}".format(newPos))
        if newPos[0] + boxSize[0] > img.shape[0] or newPos[1] + boxSize[1] > img.shape[1]:
            # clip the ROI to the image size so it fits in the remaining space
            print("Clipping ROI to fit in the image")
            newRoiSizeX = min(boxSize[0], img.shape[0] - newPos[0])
            newRoiSizeY = min(boxSize[1], img.shape[1] - newPos[1])
            
            newImg[newPos[0]:newPos[0] + boxSize[0], newPos[1]:newPos[1] + boxSize[1]] = \
                img[positions[i][0][0]:positions[i][0][0] + newRoiSizeX, \
                positions[i][0][1]:positions[i][0][1] + newRoiSizeY]
        else:
            newImg[newPos[0]:newPos[0]+boxSize[0], \
                newPos[1]:newPos[1]+boxSize[1]] = \
                img[positions[i][0][0]:positions[i][1][0], positions[i][0][1]:positions[i][1][1]]
        
    return newImg


# the function that checks if a given rectangle overlaps with any of the other rectangles
def checkOverlap(newPos: Tuple[int, int], boxSize: Tuple[int, int], positions) -> bool:
    '''
    newPos : tuple
        a tuple of two integers, X and Y of top left corner
    positions : list
        a list of tuples, first tuple - X and Y of top left corner, 
        second tuple - X and Y of bottom right corner
    '''
    newPosTL = newPos
    newPosBR = (newPos[0]+boxSize[0], newPos[1]+boxSize[1])
    # check if it overlaps with any of the other rectangles that have been drawn already
    # if it does, generate another random position and check again until it does not overlap
    for i in range(len(positions)):
        # pdb.set_trace()
        print(newPosBR, newPosTL, positions[i])
        if newPosBR[0] > positions[i][0][0] and newPosTL[0] < positions[i][1][0] and \
            newPosTL[0] > positions[i][1][1] and newPosBR[1] < positions[i][0][1]:
                return True
    return False


class ShuffleCrops():
    def __init__(self, **kwargs):
        # super().__init__(**kwargs)
        pass        
            

    def __call__(self, sample):
        sampleT = sample.permute(1,2,0)
        # sampleT = sample
        positions, _ = getIsles(sampleT)
        new_sample = torch.Tensor(drawRectangles(sampleT, positions))
        # permute back to the original shape
        new_sample = new_sample.permute(2,0,1)
        return new_sample
    








