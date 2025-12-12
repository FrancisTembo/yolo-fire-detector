from ultralytics import YOLO
import argparse

def build_model(model: str = "yolo11s"):
    """Build and load YOLOv11 model.
    
    Returns
    -------
    YOLO
        Initialised YOLOv11 model with pretrained weights.
    """
    return YOLO(f"{model}.yaml").load(f"{model}.pt")


def train(model: YOLO, data: str, hyp: str):
    """Train the YOLO model with specified configuration.
    
    Parameters
    ----------
    model : YOLO
        The YOLO model instance to train.
    data : str
        Path to the dataset configuration file.
    hyp : str
        Path to the hyperparameters configuration file.
        
    Returns
    -------
    object
        Training results object containing metrics and paths.
    """
    return model.train(data=data, cfg=hyp)



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training configuration.
    
    Returns
    -------
    argparse.Namespace
        Parsed arguments containing model, data and hyp paths.
    """
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    parser.add_argument("--model", type=str, default="yolo11s", help="YOLO model to use (e.g., yolo11n, yolo11s)")
    parser.add_argument("--data", type=str, default="config.yaml", help="Path to dataset config file")
    parser.add_argument("--hyp", type=str, default="hyp.yaml", help="Path to hyperparameters config file")
    return parser.parse_args()


def main():
    """Main function to execute the training pipeline.

    Parses arguments, builds the model, and runs training with the specified configuration.
    """
    args = parse_args()
    model = build_model(args.model)
    train(model, args.data, args.hyp)

if __name__ == "__main__":
    main()
