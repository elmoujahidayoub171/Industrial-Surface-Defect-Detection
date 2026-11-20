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

    # iterate through products
    for product in sorted(os.listdir(cwd)):

        product_path = os.path.join(cwd, product)

        if os.path.isdir(product_path):

            product_dir_path = os.path.join(product_path, "test")

            if os.path.isdir(product_dir_path):
                images = []
                labels = []

                for crash_type in sorted(os.listdir(product_dir_path)) :

                    crash_type_dir = os.path.join(product_dir_path, crash_type)
                    
                    # label the dataset
                    product_label = 0 if crash_type == "good" else 1

                    if os.path.isdir(crash_type_dir) :

        
                        for instance in sorted(os.listdir(crash_type_dir)):
                            # transform png images to np.array
                            if instance.lower().endswith(".png"):
                                ## instance here contains just the filename
                                image_path = os.path.join(crash_type_dir, instance)
                                ## pass the path of the image to the func
                                image = transform_to_array(image_path)

                                images.append(image)
                                labels.append(product_label)

                if images :
                            
                    dataset[product] = {
                        "X" : np.stack(images),
                        "y" : np.stack(labels , dtype=np.int32)
                    }

    return dataset


if __name__ == "__main__":

    dataset = load_as_numpy()

    for product, data in dataset.items():

        X = data["X"]
        y = data["y"]

        print(f"\nProduct: {product}")
        print(f"Images shape: {X.shape}")
        print(f"Labels shape: {y.shape}")

        print(
            "Label distribution:",
            np.unique(y, return_counts=True)
        )


