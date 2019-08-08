# LookingForSeagrasLostPipeline

#Instruction

# Seagrass LOST Pipeline
This Repo is a setting up example for semantic Segmentation in LOST.
This Repo is based on this Repo ([Link](https://github.com/EnviewFulda/LookingForSeagrassSemanticSegmentation)) 
## Instruction
1. Install LOST ([Link](https://github.com/l3p-cv/lost))
2. Clone this Repository into ```{LOST WORKSPACE}/my_data/{USER}```
3. Clone the branch "LostPipeline" from this Repository: ([Link](https://github.com/EnviewFulda/LookingForSeagrassSemanticSegmentation)) into the project Folder
2. Download the Dataset ([Link](https://drive.google.com/drive/folders/1X0pmRIkPRC672_vuWqotfLdgbHx1QpFZ))
3. Unzip the Dataset
4. ```cd dataset```
5. ```mkdir Seagrass_Flat```
6. Flat the images with ```cp images/*/* ./Seagrass_Flat/```
7. ```mkdir gt_flat```
8. Flat the images with ```cp ground-truth/*/* ./gt_flat/```
9. Copy the Folder ```Seagrass_Flat/``` into  ```{LOST WORKSPACE}/data/media```
10. Copy the Folder ```gt_flat/``` into  ```{LOST WORKSPACE}/my_data/{USER}```
11. import the pipeline ([Link](https://lost.readthedocs.io/en/latest/lost_cli.html?highlight=import#managing-pipeline-projects))
12. Add a Label Tree "Seagrass" with the Labels "Seagrass" and "Background"
13. Start the Pipeline 

