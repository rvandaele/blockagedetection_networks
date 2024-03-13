# blockagedetection_networks

This code accompanies our paper currently under revision: Vandaele R., Ojha V., Dance S.-L., Deep Learning for Automated Trash Screen Blockage Detection Using Cameras: Actionable Information for Flood Risk Management, Journal of Hydroinformatics

The python code allows user to detect blockage on new trash screen images using the networks developed during our publication. For the moment, this code does not allow to train the networks.
- classification_network.py corresponds to the classification approach
- padim.py corresponds to the anomaly detection approach
- siamese_network.py corresponds to the Siamenese network approach

The *crop_coordinates.txt* file contains the coordinates of the windows we manually delineated for each trash screen camera used for our publication.

The corresponding dataset (images and optimized network weights) can be found at: https://researchdata.reading.ac.uk/498/
- The test cameras used in our publication were *Cornwall_Crinnis, Cornwall_MevagisseyPreScreen, Devon_BarnstapleBradiford* and *sites_sistontunnel_cam1*
- The networks have not been trained on the test cameras

**Requirements** 
	- numpy (tested with version 1.24.3)
	- pytorch (tested with version 2.1.0)
	- torchvision (tested with version 0.15.2)
	
**How to use**

Let's say that the dataset obtained at https://github.com/rvandaele/blockagedetection_networks has been extracted at the location */home/user/blockagedetection_dataset*

In the code, the 
- the model filepaths variable must be assigned to the corresponding network weights (e.g., 
	- *model_filepath* must be assigned to */home/user/blockagedetection_dataset/weights/classification.pth* for *classification_network.py* and to */home/user/blockagedetection_dataset/weights/siamese.pth* for *siamese_network.py*
	- *padim_mean_fpath* must be assigned to */home/user/blockagedetection_dataset/weights/padim_resnet_mean_4.pth* and *padim_cov_fpath* must be assigned to */home/user/blockagedetection_dataset/weights/padim_resnet_cov_4.pth*
- the *image_filenames* variable must be assigned to a list of images for which the user wants an estimation, e.g ["/home/user/blockagedetection_dataset/images/Cornwall_Crinnis/clear/2022_01_28_15_07.jpg", "/home/user/blockagedetection_dataset/images/Cornwall_Crinnis/blocked/2022_02_08_16_08.jpg"]
- the x_min, x_max, y_min, y_max can be assigned to select a region of interest within the trash screen image (see *crop_coordinates.txt* for the values that were used in our work, leave to -1 if you don't want to use any window)
- For *siamese_network.py*, you also need to mention a list of clear reference images by assigning the variable *clean_ref_imgs*, e.g ["/home/user/blockagedetection_dataset/images/Cornwall_Crinnis/clear/2022_03_26_06_10.jpg", "/home/user/blockagedetection_dataset/images/Cornwall_Crinnis/clear/2022_03_28_13_10.jpg"]

You can then launch the evaluation script in command line using python. The script will return blockage scores and classes for each image within the *image_filenames* list