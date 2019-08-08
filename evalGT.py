from lost.pyapi import script
from LookingForSeagrassSemanticSegmentation import main
import os
import json
import pickle
import time
ENVS = ['lost-cv-gpu','lost-cv']
RESOURCES = ['lock_all']

ARGUMENTS = {'Ground_Trouth_result_name' : { 'value':'GT_results.json',
                            'help': 'Grouth Truth evaluation results'}
            }

class EvalGT(script.Script):
    def main(self):
        shared_variables_path = self.get_path("shared_variables.p", context="pipe")
        shared_variables = pickle.load( open( shared_variables_path, "rb" ) )
        self.logger.info("PIPE IS RUNNING")
        self.logger.info(shared_variables['pipe_is_running'])
        if(shared_variables['pipe_is_running']):
            models_path = self.get_path('model/', context='pipe')
            eval_result_folder_path = self.get_path('eval_result_GT/')
            if not os.path.exists(eval_result_folder_path):
                os.mkdir(eval_result_folder_path)
            GT_result_path = eval_result_folder_path + self.get_arg('Ground_Trouth_result_name')
            labeled_data_path = self.get_path("gt_flat/", context='static')
            meanIoU, eval_status = main.deepSS(
                "eval","deeplabV3plusSS", 
                scr=self,
                models_path= models_path,
                model_file_name = 'seagrass.ckpt',
                images_path = '/home/lost/data/media/Seagrass_Flat/',
                labeled_images_path = labeled_data_path,
                train_json_path = self.get_path('train.json', context='pipe'),
                eval_json_path = self.get_path('jsonData/eval.json', context ='static'),
                config_path = self.get_path("jsonData/configDataSeagrass.json", context='static'),
                net_config_path= self.get_path("jsonData/deeplabV3plusSSConfig.json", context='static'),
                iteration= self.iteration
                )
            if os.path.exists(GT_result_path):
                with open(GT_result_path, 'r') as outfile:
                    results = json.load(outfile)
            else:
                results = []
            results.append(eval_status)
            with open(GT_result_path, 'w') as outfile:
                json.dump(results, outfile, indent=1, ensure_ascii=True)
            self.outp.add_data_export(file_path=GT_result_path)
if __name__ == "__main__":
    my_script = EvalGT() 
