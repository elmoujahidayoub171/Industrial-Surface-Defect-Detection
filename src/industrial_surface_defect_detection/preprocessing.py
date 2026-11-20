## import dependencies
import numpy as np 
from sklearn.feature_extraction import image
import os
import cv2
import matplotlib.pyplot as plt


def transform_to_array(image_path) -> np.ndarray :
    # by default the image is BGR
    image = cv2.imread(image_path,
                       cv2.IMREAD_UNCHANGED)

    if image is None :
        raise FileNotFoundError(f"cannot read image : {image_path}")

    # transform to RGB
    image  = cv2.cvtColor(image ,
                          cv2.COLOR_BGR2RGB)

    # resize image
    image = cv2.resize(image , (256 , 256) ,interpolation=cv2.INTER_AREA)

    # normalization
    image = image.astype(np.float32) / 255

    return image


def load_as_numpy() -> dict:
    ## go to the root path
    os.chdir("/home/umoujahid/Industrial_Surface_Defect_Detection/MVTec_AD/")

    cwd = os.getcwd()

    dataset = {}

    print("Loading dataset ...")

    # iterate through products
    for product in sorted(os.listdir(cwd)):

        product_path = os.path.join(cwd, product)

        if os.path.isdir(product_path):

            product_dir_path = os.path.join(product_path, "train", "good")

            if os.path.isdir(product_dir_path):
                images = []
        
                for instance in sorted(os.listdir(product_dir_path)):
                        # transform png images to np.array
                        if instance.lower().endswith(".png"):
                            ## instance here contains just the filename
                            image_path = os.path.join(product_dir_path, instance)
                            ## pass the path of the image to the func
                            image = transform_to_array(image_path)


                            images.append(image)

                if images :
                    dataset[product] = np.stack(images)
    print("Data Loaded successfully !")

    return dataset

# save preprocessed data and separate products
def save_processed_dataset(dataset : dict) -> None :
    output_dir = "/home/umoujahid/Industrial_Surface_Defect_Detection/MVTec_AD_Preprocessed/train/preprocessed"
    
    #check if the output_dir is created 
    os.makedirs(output_dir , exist_ok=True)

    print("Saving products images as np arrays ...")

    # save products 
    for product , images in dataset.items():
        save_path = os.path.join(output_dir , f"{product}.npy")
        
        np.save(save_path, images)

    print(f"Dataset saved to {output_dir} !")

    

if __name__ == "__main__" :
    dataset = load_as_numpy()
    save_processed_dataset(dataset)

    



