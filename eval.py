from ultralytics import YOLO
import argparse


def load_model(model: str):
    """Load a trained YOLO model for evaluation.
    
    Parameters
    ----------
    model : str
        Path to the trained model weights file.
    
    Returns
    -------
    YOLO
        Loaded YOLO model ready for evaluation.
    """
    return YOLO(model)


def evaluate(model: YOLO, data: str, cfg: str):
    """Evaluate the YOLO model on validation dataset.
    
    Parameters
    ----------
    model : YOLO
        The YOLO model instance to evaluate.
    data : str
        Path to the dataset configuration file.
    cfg : str
        Path to the evaluation configuration file.
        
    Returns
    -------
    object
        Evaluation results object containing metrics.
    """
    return model.val(data=data, cfg=cfg)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation configuration.
    
    Returns
    -------
    argparse.Namespace
        Parsed arguments containing model, data and cfg paths.
    """
    parser = argparse.ArgumentParser(description="YOLO Evaluation Script")
    parser.add_argument("--model", type=str, help="Path to trained YOLO model weights")
    parser.add_argument("--data", type=str, default="config.yaml", help="Path to dataset config file")
    parser.add_argument("--cfg", type=str, default="eval.yaml", help="Path to evaluation config file")
    return parser.parse_args()


def main():
    """Main function to execute the evaluation pipeline.

    Parses arguments, loads the model, and runs evaluation with the specified configuration.
    """
    args = parse_args()
    model = load_model(args.model)
    evaluate(model, args.data, args.cfg)


if __name__ == "__main__":
    main()