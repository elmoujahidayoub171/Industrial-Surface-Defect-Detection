from sklearn.metrics import ( accuracy_score , 
                                precision_score ,
                                recall_score ,
                                f1_score ,
                                roc_auc_score ,
                                average_precision_score,
                                confusion_matrix,
                                ConfusionMatrixDisplay,
                                RocCurveDisplay,
                                PrecisionRecallDisplay,
                                                       
)
import matplotlib.pyplot as plt
import numpy as np
import json
import os


def evaluate_model(model , X_test : np.ndarray , y_test : np.ndarray) -> dict :
    ## binary prediction 
    # 0 :  normal , 1 :  anormal
    y_pred = model.predict(X_test)

    ## anomaly score
    anomaly_scores = model.anomaly_score(X_test)

    ## evaluation_metrics
    metrics ={
        "Accuracy" :  accuracy_score(y_test, y_pred) ,
        "Precision" : precision_score(y_test, y_pred , zero_division=0) , # to avoid error when 0 appears in denominator , returns 0 instead error
        "Recall" : recall_score(y_test, y_pred , zero_division=0),
        "F1 score" : f1_score(y_test, y_pred, zero_division=0) , 
        "ROC_AUC" :  roc_auc_score(y_test , anomaly_scores),
        "Average precision" : average_precision_score(y_test, anomaly_scores),
    }

    ## Confusion matrix report
    cm = confusion_matrix(y_test, y_pred , labels=[0,1])
    tn = cm[0,0]
    fp=cm[0,1]
    fn= cm[1,0]
    tp=cm[1,1]


    # calculate  FPR & TPR
    FPR = fp / (fp + tn) if (fp + tn) != 0 else 0
    TPR = tp / (tp + fn) if (tp + fn) !=0 else 0

    metrics["FPR"] , metrics["TPR"] = FPR , TPR


    ## display results 
    print("\t Model Evaluation")

    for metric , value in metrics.items():
        print(f"{metric :<20} : {value:.4f}")

    # confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Normal" , "Deficitve"],
        cmap="Blues"
    )

    plt.title("Confusion Matrix")
    plt.show()

    # roc curve
    RocCurveDisplay.from_predictions(
        y_test,
        anomaly_scores
    )
    plt.title("Roc Curve")
    plt.show()

    # precision-recall curve 
    PrecisionRecallDisplay.from_predictions(
        y_test ,
        anomaly_scores
    )

    plt.title("PR curve")
    plt.show()

    # Anomaly Score distribution
    plt.figure(figsize=(9,10))

    plt.hist(
        anomaly_scores[y_test ==  0 ],
        bins=20 ,
        alpha=0.6,
        label = "Normal"
    )

    plt.hist(
        anomaly_scores[y_test ==  1 ],
            bins=20 ,
            alpha=0.6,
            label = "Defictive"
    )

    plt.xlabel("Anomaly score")
    plt.ylabel("Number of images")
    plt.title("Anomaly score distribution")
    plt.legend()
    plt.show()

    return metrics


def save_metrics(all_metrics : dict ) -> None :
    ## path where saving the metrics
    results_path = "/home/umoujahid/Industrial_Surface_Defect_Detection/metrics/evaluation_metrics.json"
    os.makedirs(os.path.dirname(results_path) , exist_ok=True)

    ##Convert numpy metrics to python float
    serializable_metrics = {
                product : {
                            metric : float(value)
                            for metric , value in metrics.items()
                            }
                        for product , metrics in all_metrics.items()
    }            

    ## saving 
    with open(results_path , 'w' , encoding='utf-8') as f :
        json.dump(serializable_metrics , f , indent=4)

    print(f"Metrics saved to  : {results_path}")
    

if __name__ == "__main__" :
    ...




    









