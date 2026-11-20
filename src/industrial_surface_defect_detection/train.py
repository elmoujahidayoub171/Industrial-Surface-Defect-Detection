from industrial_surface_defect_detection.model import Model
import numpy as np 
import os
from pathlib import Path


def main(preprocessed_path : str , output_dir : str ): ## here preprocessed path point to the folder which
                                                                #contains the features dataset
    # load data 
    preprocessed_path = Path(preprocessed_path)
    output_dir = Path(output_dir)

    os.makedirs(output_dir , exist_ok=True)

    for category in sorted(os.listdir(preprocessed_path)) :
        
        if not category.lower().endswith(".npy") :
            continue

        product = category.removesuffix(".npy")
        print(f'Processing product : {product}')
        
        category_path = os.path.join(preprocessed_path , category)
        X_features = np.load(category_path)

        print(f"Images Loaded : {X_features.shape}")
        
        ## initialize the model
        model = Model(
            n_estimators=100 ,
            contamination="auto" ,
            random_state = 42 ,
        )

        ## train
        print("Starting training ...")

        model.fit(X_features)

        ## save trained model
        save_path = os.path.join(output_dir ,  f"{product}_isolation_forest.joblib")
        model.save(save_path)
        print(f"Model saved to {save_path}")

    print("All models trained successfully .")

if __name__ == "__main__" :
    preprocessed_path = Path("/home/umoujahid/Industrial_Surface_Defect_Detection/MVTec_AD_Preprocessed/train/features/")
    output_dir = Path("/home/umoujahid/Industrial_Surface_Defect_Detection/models/")

    main(preprocessed_path , output_dir)
    






