from lost.pyapi import script
import os
import random
import json
import time
from LookingForSeagrassSemanticSegmentation import main
import cv2
import polygon_transform
import numpy as np
import pickle

ENVS = ['lost-cv-gpu','lost-cv']
RESOURCES = ['lock_all']


ARGUMENTS = {'n' : { 'value': 150,
                    'help': 'Number of images that will be request during each iteration.'
                    },
            'trainsize' : {'value': 0.8,
            'help': 'n* trainsize gives the number of Training Images (must be < 1). The rest are the evaluation Images'
            }
            }

class RequestLoopAnnos(script.Script):
    '''Annotations in a loop setup.
    '''
    def main(self):
        self.logger.info("---------------------------------------------------------------------")
        self.logger.info(str(self.iteration))
        self.logger.info("---------------------------------------------------------------------")
        for ds in self.inp.datasources:
            media_path = ds.path

            models_path = self.get_path('model/', context='pipe')
            train_path = self.get_path('train.json', context='pipe')
            eval_path = self.get_path('eval.json', context='pipe')

            raw_train_path = self.get_path('jsonData/train.json', context = 'static')
            raw_eval_path = self.get_path('jsonData/eval.json', context = 'static')
            raw_train_json = []
            raw_eval_json = []

            result_predict_path = self.get_path('images_predicted/')
            if(not os.path.exists(result_predict_path)):
                os.mkdir(result_predict_path)

            trainsize = float(self.get_arg('n')) * float(self.get_arg('trainsize'))
            trainsize = int(round(trainsize))

            evalsize_float = 1.0- float(self.get_arg('trainsize'))
            evalsize = float(self.get_arg('n')) * evalsize_float
            evalsize = int(round(evalsize))
            
            with open(raw_train_path, 'r') as f:
                raw_train_json= json.load(f)
            with open(raw_eval_path, 'r') as f:
                raw_eval_json = json.load(f)

            start_train_index = self.iteration * trainsize
            end_train_index = start_train_index+ trainsize
            train_list = raw_train_json[:end_train_index]
            
            start_eval_index = self.iteration*evalsize
            end_eval_index = start_eval_index + evalsize
            eval_list = raw_eval_json[:end_eval_index]
            # only last elements
            anno_list = train_list[start_train_index:] + eval_list[start_eval_index:]
            
            shared_variables_path = self.get_path("shared_variables.p", context="pipe")
            shared_variables = { "pipe_is_running": True }
            pipe_is_running = True
            if(len(anno_list) < (trainsize  + evalsize)):
                shared_variables = { "pipe_is_running": False }
                self.break_loop()
                pipe_is_running = False
                picture_end_path = self.get_path("ende_image.jpg", context="static")
                self.outp.request_annos(
                    img_path= picture_end_path
                )
            pickle.dump( shared_variables, open( shared_variables_path, "wb" ) )
            if(pipe_is_running):
                if(self.iteration == 0):
                    for i, anno in enumerate(anno_list):
                        self.update_progress((i/len(anno_list))*100)
                        img_name = anno['image'].split('/')[-1]
                        img_path = os.path.join(media_path, img_name)
                        self.outp.request_annos(img_path=img_path)
                else:
                    predicted_images = main.deepSS("predict", "deeplabV3plusSS",
                        scr=self,
                        images_to_predict = anno_list,
                        models_path= models_path,
                        model_file_name = 'seagrass.ckpt',
                        images_path = '/home/lost/data/media/Seagrass_Flat/',
                        train_json_path = self.get_path('train.json', context='pipe'),
                        eval_json_path = self.get_path('eval.json', context ='pipe'),
                        config_path = self.get_path("jsonData/configDataSeagrass.json", context='static'),
                        net_config_path= self.get_path("jsonData/deeplabV3plusSSConfig.json", context='static'),
                        result_predict_path= result_predict_path, 
                        iteration= self.iteration,
                        media_path = media_path,

                    )

                    for i, pixel_map in enumerate(predicted_images):
                        img_name = pixel_map.split('/')[-1]
                        pixel_map = cv2.imread(pixel_map)
                        polygons, anno_types, anno_labels  =  polygon_transform.pixelMapToPolygons(pixel_map, self)
                        # get image and add polyon to the image
                        img_name = anno_list[i]['image'].split('/')[-1]
                        img_path = os.path.join(media_path, img_name)
                        self.outp.request_annos(
                            img_path=img_path, 
                            anno_labels=anno_labels, 
                            annos=polygons, 
                            anno_types=anno_types)
        with open(train_path, 'w') as f:
            json.dump(train_list, f)
        with open(eval_path, 'w') as f:
            json.dump(eval_list, f)

if __name__ == "__main__":
    my_script = RequestLoopAnnos()