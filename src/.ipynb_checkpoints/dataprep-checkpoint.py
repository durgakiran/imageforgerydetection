import numpy as np
from os import listdir, path
import cv2
from src.utils import ImageOperations
from random import shuffle
from shutil import copy



class DataPrep():
    def __init__(self):
        super()
    
    def splitData(self, source: str, training: str, validation: str, test: str, test_split_size: float = 0.1, validation_split_size: float = 0.1):
        files = []
        imageOp = ImageOperations()
        for f in listdir(source):
            file = source + f
            if imageOp.isImageExtension(f) and path.getsize(file) > 0:
                files.append(f)
            else:
                print('{} is zero sized or not an image'.format(f))
        
        training_length = int(len(files) * (1 - (test_split_size + validation_split_size)))
        validation_length = int(len(files) * validation_split_size)
        test_length = int(len(files) * test_split_size)
 
        #shuffle files randomly
        shuffle(files)

        #divide the shuffled files into test, train, validation
        training_set =  files[: training_length]
        validation_set = files[training_length: (training_length + validation_length)]
        test_set = files[(training_length + validation_length) : ]

        print(training_length, validation_length, test_length)

        for f in training_set:
            tmp_file = source + f
            copy(tmp_file, training)
        
        for f in validation_set:
            tmp_file = source + f
            copy(tmp_file, validation)
        
        for f in test_set:
            tmp_file = source + f
            copy(tmp_file, test)
        
        print('*********Data splitting done************')
    

    def getBoundingBoxes(self, img_path: str, mask_path: str):
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)

        # converting image to binary
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
        mask_grey = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) # this would convert the 3 channel image to single channel
        blur = cv2.GaussianBlur(mask_grey,(5,5),0) # to remove any noise present
        ret3,th3 = cv2.threshold(blur,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # ostru binary thresholding

        # finding coordinates with pixel value 0
        coords = np.where(th3 == 1)
        
        row_0 = min(coords[0])
        row_1 = max(coords[0])
        col_0 = min(coords[1])
        col_1 = max(coords[1])

        # adjusting image bounding box coordinates
        if(row_0 >= 64):
            row_0_adjusted = row_0 - 64
        elif (row_0 >= 0 and row_0 < 64):
            row_0_adjusted = 0
        
        if((mask.shape[0] - row_1) >= 64):
            row_1_adjusted = row_1 + 64
        elif (row_1 <= mask.shape[0] and (mask.shape[0] - row_1) < 64):
            row_1_adjusted = mask.shape[0]

        if(col_0 >= 64):
            col_0_adjusted = col_0 - 64
        elif (col_0 >= 0 and col_0 < 64):
            col_0_adjusted = 0

        if((mask.shape[1] - col_1) >= 64):
            col_1_adjusted = col_1 + 64
        elif (col_1 <= mask.shape[1] and (mask.shape[1] - col_1) < 64):
            col_1_adjusted = mask.shape[1]

        new_mask = th3[row_0_adjusted: row_1_adjusted, col_0_adjusted:col_1_adjusted]
        
        new_img = img[row_0_adjusted: row_1_adjusted, col_0_adjusted:col_1_adjusted, :]

        return new_mask, new_img


    def ExtractPatches(self, img_path, mask_path):
        patches = []
        new_mask, new_img = self.getBoundingBoxes(img_path, mask_path)
        (h, w, c) = new_img.shape
        h_range = h - 64
        w_range = w - 64
        for i in range(0,h_range, 2):
            for j in range(0, w_range, 2):
                patch = np.zeros((64,64,3))
                patch_mask = np.zeros((64,64))
                for k in range(64):
                    row_mask = new_mask[(i + k):(i + 1 + k), ( j ) : (j+  64)]
                    row = new_img[(i + k):(i + 1 + k), ( j ) : (j+ 64), :]
                    patch[k, :, :] = row
                    patch_mask[k, :] = row_mask
                
                patch = np.array(patch, dtype='int')
                patch_mask = np.array(patch_mask, dtype='int')
                if (np.sum(patch_mask) >= 0.30*64*64 
                                and (np.sum(patch_mask) < 0.90*64*64)):
                    patches.append(patch)
                else:
                    continue
        if len(patches) > 500:
            patches_index = np.random.choice(len(patches), 500, replace=False)
            patch_arr = [patches[q] for q in patches_index]
        else:
            patches_index = range(len(patches))
            patch_arr = patches

        return patches_index, patch_arr
    
    def extractPristinePatches(self, img_path, mask_path, threshold = 500, stride = (10, 10), size = (64, 64), padding='VALID'):
        patches = []
        new_img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        mask_grey = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) # this would convert the 3 channel image to single channel
        blur = cv2.GaussianBlur(mask_grey,(5,5),0) # to remove any noise present
        ret3, new_mask = cv2.threshold(blur,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # ostru binary thresholding
        
        (h, w, c) = new_img.shape
        h_range = h - 64
        w_range = w - 64
        # print(h_range, w_range)
        for i in range(0,h_range, 10):
            for j in range(0, w_range, 10):
                patch = np.zeros((64,64,3))
                patch_mask = np.zeros((64,64))
                for k in range(64):
                    
                    # print(new_img.shape)
                    row_mask = new_mask[(i + k):(i + 1 + k), ( j ) : (j+  64)]
                    row = new_img[(i + k):(i + 1 + k), ( j ) : (j+ 64), :]
                    patch[k, :, :] = row
                    patch_mask[k, :] = row_mask
            
            patch = np.array(patch, dtype='int')
            patch_mask = np.array(patch_mask, dtype='int')
            if (np.sum(patch_mask) <= 1*64*64 and (np.sum(patch_mask) > 0*64*64)):
                continue
            else:
                patches.append(patch)
        if len(patches) > 800:
            patches_index = np.random.choice(len(patches), 500, replace=False)
            patch_arr = [patches[q] for q in patches_index]
        else:
            patches_index = range(len(patches))
            patch_arr = patches

        return patches_index, patch_arr
    