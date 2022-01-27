# Pedestrian Attribute Recognition

Requirements:    
1. Python    
2. Pytorch  
<br>


Steps to run the code:    
1. Install anaconda    
2. Open conda prompt and create conda virtual envirnoment    
    2.1 Running the below command in prompt   
        ```conda env create --name open_cv --file sw/attribute_classify.yml ```   
        This will create virtual environment with name open_cv using environment requirements dependincies mentioned in sw/attribute_classify.yml file.    
    2.2 Switch from base to open_cv environment by typing.    
        ```conda activate open_cv``` 
3. Once conda environment is activated.   
    3.1 Run ```python sw/transform_peta.py``` to perform image operation and spltting data in training and testing set. This will also generate 2 pkl files in PETA dataset folder that will be required for model training and testing.    
    3.2 Run ```sh sw/train.sh``` to perform model training.    
    3.3 Run ```sh sw/test.sh``` to perform model testing.    
    3.4 Once model is trained or using downloaded pretrained model, run ```python sw/demo.py``` to generate a demo output predictions.    
    3.5 Using sw/example.mp4 video we can visualize the working of model. Run command ```python sw/person_detect.py```, it will generate an output video file performing predictions of attributes and also saving the output file.



    
