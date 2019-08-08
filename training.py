from lost.pyapi import script
import os
import random
import cv2
import numpy as np
from skimage.measure import find_contours, approximate_polygon, \
    subdivide_polygon

from LookingForSeagrassSemanticSegmentation import main
import json
import pickle
import polygon_transform


# train = 70 prozent
# test = 20 prozent
# validate = 10 prozent




ENVS = ['lost-cv-gpu','lost-cv']
RESOURCES = ['lock_all']
ARGUMENTS = {'SIA_result_name' : { 'value':'SIA_result.json',
                            'help': 'Grouth Truth evaluation results'},
            'SIA_best_epochs_name' : { 'value':'SIA_best_epochs.json',
                            'help': 'Grouth Truth evaluation results'}
            }

# np.set_printoptions(threshold=np.inf)
class Train(script.Script):
    '''Request annotations for each image of an imageset.

    An imageset is basicly a folder with images.
    '''
    def main(self):
        shared_variables_path = self.get_path("shared_variables.p", context="pipe")
        shared_variables = pickle.load( open( shared_variables_path, "rb" ) )
        self.logger.info("PIPE IS RUNNING")
        self.logger.info(shared_variables['pipe_is_running'])
        if(shared_variables['pipe_is_running']):
            models_path = self.get_path('model/', context='pipe')
            eval_result_path = self.get_path('eval_result_SIA/')
            labeled_data_path = self.get_path("images_labeled/")
            if not os.path.exists(models_path):
                os.mkdir(models_path)
            if not os.path.exists(eval_result_path):
                os.mkdir(eval_result_path)
            if not os.path.exists(labeled_data_path):
                os.mkdir(labeled_data_path)


        #     # START GENERATE PIXELMAP IMAGES
        #     df = self.inp.to_df()
        #     labeled_data_path = self.get_path("images_labeled/")
        #     ### FOR TEST
        #     labeled_data_path = self.get_path("gt_flat/", context ="static")
        #     if not os.path.exists(labeled_data_path):
        #         os.mkdir(labeled_data_path)
        #     for i, img in enumerate(self.inp.img_annos): 
        #         path = self.get_abs_path(img.img_path)
        #         image = cv2.imread(path)
        #         height, width = image.shape[:2]
        #         mult = np.array([width, height])
        #         # init with background
        #         polygon_image = np.full((height, width), 255.0)
        #         # last element
        #         ground_truth_image_name = "pm_" + img.img_path.split('/')[-1]
        #         # replace img with png
        #         ground_truth_image_name = ground_truth_image_name.replace('jpg', "png")

        #         ground_truth_path = labeled_data_path + "/" + ground_truth_image_name
        #         # FOR TEST ONLY
        #         for i, anno in enumerate(img.twod_annos):
        #             anno = anno.to_vec(['anno.data', 'anno.lbl.idx', 'anno.lbl.name'])
        #             anno_data = anno[0]
        #             label = anno[2]
        #             if (label == 'Background') and (i == 0):
        #                 polygon_image = np.full((height, width), 0.0)
        #             # muliply with the height and widht of the picture to get the exakt position of the polygon
        #             anno_data = np.array(anno_data) * mult
        #             anno_data = [np.array(anno_data, dtype=np.int32 )]
        #             # 255 is white; white means Background
        #             # 0 is black; Black means Seagrass
        #             color = 255.0
        #             if(label =="Seagrass"):
        #                color = 0.0
        #             # pixelmap to polygon
        #             cv2.fillPoly(polygon_image, anno_data, color)
        #         cv2.imwrite(ground_truth_path,polygon_image) 
        # # END GENERATE PIXELMAP IMAGES





            for i, img in enumerate(self.inp.img_annos): 
                       path = self.get_abs_path(img.img_path)
                       image = cv2.imread(path)
                       img_height, img_width = image.shape[:2]
                       # mult = np.array([img_width, img_height])
                       polygon_list = []
                       for i, anno in enumerate(img.twod_annos):
                           anno = anno.to_vec(['anno.data', 'anno.lbl.idx', 'anno.lbl.name'])
                           # anno_data = np.array(anno_data) * mult
                           # anno_data = [np.array(anno_data, dtype=np.int32 )]
                           polygon_list.append(anno)
                       pixel_map = polygon_transform.polygonsToPixelMap(polygon_list, img_height,img_width, self)
                       ground_truth_image_name = "pm_" + img.img_path.split('/')[-1]
                       ground_truth_image_name = ground_truth_image_name.replace('jpg', "png")
                       ground_truth_path = labeled_data_path + "/" + ground_truth_image_name
                       cv2.imwrite(ground_truth_path,pixel_map)





            SIA_result_path = eval_result_path + self.get_arg('SIA_result_name')
            SIA_best_epochs_path = eval_result_path + self.get_arg('SIA_best_epochs_name')
        # START TRAINING
            main.deepSS(
                "train","deeplabV3plusSS", 
                scr=self,
                models_path= models_path,
                model_file_name = 'seagrass.ckpt',
                images_path = '/home/lost/data/media/Seagrass_Flat/',
                labeled_images_path = labeled_data_path,
                train_json_path = self.get_path('train.json', context='pipe'),
                eval_json_path = self.get_path('eval.json', context ='pipe'),
                config_path = self.get_path("jsonData/configDataSeagrass.json", context='static'),
                net_config_path= self.get_path("jsonData/deeplabV3plusSSConfig.json", context='static'),
                iteration= self.iteration,
                SIA_result_path = SIA_result_path,
                SIA_best_epochs_path = SIA_best_epochs_path
                )
            self.outp.add_data_export(file_path=SIA_result_path)
            self.outp.add_data_export(file_path=SIA_best_epochs_path)

        # END TRAINING

if __name__ == "__main__":
    my_script = Train() 











