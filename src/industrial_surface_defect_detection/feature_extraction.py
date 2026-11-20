import numpy as np
import cv2
import os 
from skimage.feature import local_binary_pattern

# func that build feature vector
def extract_feature(image : np.ndarray) -> np.ndarray :

    ##define hog descriptor
    hog_descriptor = cv2.HOGDescriptor(
        (256,256), #window size
        (32,32), #Block size
        (16,16), # block stride
        (16,16), # HOG cell size
        9 , # Number of orientation bins
    )

    ## define lbp descriptor

    # transform RGB image to uint8
    uint8_image = (image * 255.0).astype("uint8")

    # convert to grayscale
    gray = cv2.cvtColor(uint8_image,
                        cv2.COLOR_RGB2GRAY)

    # cmpute HOG 
    hog_features = hog_descriptor.compute(gray).ravel() ##flattening the features vector

    # compute_lBP
    #lbp parameters
    P = 8
    R = 1
    lbp = local_binary_pattern(
        gray , # 2D grayscale image
        P=P, # int number of symetric neighbors
        R=R, # radius of circle
        method="uniform"
    )

    # extract normalized histogram
    n_bins = P + 2

    lbp_histogram , _ = np.histogram(
        lbp.ravel() ,
        bins=n_bins ,
        range=(0, n_bins)
    )

    lbp_histogram = lbp_histogram.astype(np.float32)
    lbp_histogram /= lbp_histogram.sum()

    ## concatenate both lbp and hog patterns
    combined_features = np.concatenate([
        hog_features,
        lbp_histogram
    ])

    return combined_features


# func that build feature matrix from feature vectors
def build_feature_matrix(images : np.ndarray) -> np.ndarray :
    print("Extracting Features ...")
    features = []
    # iterate through all instaces of product
    for image in images :
        feature_vector = extract_feature(image)

        features.append(feature_vector)

    X = np.stack(features)


    return X

def build_all_feature_matrix(dataset : dict) -> dict :
    feature_dataset = {}

    for product, images in dataset.items():
        print(f"Processing {product} ...")
        X = build_feature_matrix(images)

        feature_dataset[product] = X
    return feature_dataset

# save feature matrix off each product
def save_features_dataset(dataset : dict) -> None :
        output_dir = "/home/umoujahid/Industrial_Surface_Defect_Detection/MVTec_AD_Preprocessed/train/features/"
        
        #check if the output_dir is created 
        os.makedirs(output_dir , exist_ok=True)

        print("Saving features matrix ...")

        # save products 
        for product , features in dataset.items():
                save_path = os.path.join(output_dir , f"{product}.npy")
                
                np.save(save_path, features)

        print(f"Dataset saved to {output_dir} !")


if __name__ == "__main__" :
    preprocessed_dataset_path = "/home/umoujahid/Industrial_Surface_Defect_Detection/MVTec_AD_Preprocessed/train/preprocessed/"

    dataset = {}

    for category in os.listdir(preprocessed_dataset_path) :

        if not category.lower().endswith("npy") :
            continue

        category_path = os.path.join(preprocessed_dataset_path , category)

        product = category.removesuffix('.npy') ## python method to remove the extention and keep the product name

        dataset[product] = np.load(category_path)


    ## build feauture dataset
    features_dataset = build_all_feature_matrix(dataset)

    ## save feature dataset

    save_features_dataset(features_dataset)


