"This script uses the trained model with the ultralytics tracking pipeline"


import argparse
import cv2

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11 Object Tracking")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: camera index (0, 1, etc.) or path to video file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="../weights/best.pt",
        help="Path to YOLO model weights",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the YOLO11 model
    model = YOLO(args.model)

    # Set up video capture (camera or video file)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.source}")
        return

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.track(frame, persist=True)
            annotated_frame = results[0].plot()

            cv2.imshow("YOLO11 Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()