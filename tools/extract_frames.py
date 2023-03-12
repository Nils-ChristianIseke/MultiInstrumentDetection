import argparse
import cv2 as cv2
import os
from pathlib import Path
import numpy as np


def main(args):
    assert os.path.exists(args.video), "Path to videos is not valid"
    video_name = Path(args.video).stem
    video_parent_path = Path(args.video).parent

    cap = cv2.VideoCapture(args.video)
    while True:

        # -----------------------------------------
        success, frame = cap.read()
        if not success: break

        current_frame = cap.get(1)
        current_time = cap.get(0) / 1000  # get time in ms and convert to s

        frame_height = cap.get(4)
        frame = swap_top_bottom_halves(frame, frame_height)
        # ------------------------------------------

        for next in range(args.succ):
        # routine to append successors to frame
            # -----------------------------------------
            success, frame_succ = cap.read()
            if not success: break

            frame_height = cap.get(4)
            frame_succ = swap_top_bottom_halves(frame_succ, frame_height)
            # ------------------------------------------
            frame = np.append(frame, frame_succ, axis=1)

        cap.set(1, cap.get(1) - args.succ)
        if ((args.start_seconds <= current_time <= args.end_seconds)
                and (current_frame % args.sample_freq == 0)):
            im_name = "{}_f{}.png".format(video_name, str(int(current_frame)).zfill(4))
            cv2.imwrite(os.path.join(video_parent_path, im_name), frame)

    # After extraction, then:
    cap.release()
    cv2.destroyAllWindows()


def swap_top_bottom_halves(image, height):
    top = image[:int(height/2), :]
    bottom = image[int(height/2):, :]
    return np.concatenate((bottom, top), axis=0)


def split_image_top_down(image):
    """
    :param image: np array
    :return: two np arrays corresponding to the top and bottom images
    """
    im_height = image.shape[0]
    assert im_height%2 == 0, "Image height is not even"
    top_image = image[: im_height//2, ...]
    bottom_image = image[im_height//2 :, ...]
    return top_image, bottom_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='split annotated video frames')
    parser.add_argument('-v', '--video', dest='video', required=True, type=str, help='path to video')
    parser.add_argument('--sample_freq', dest='sample_freq', type=int, default=120, help='sample x-th frame')
    parser.add_argument('--successors', dest='succ', required=True, type=int, default=0, help="no. temporal frames")
    parser.add_argument('-s', '--start_seconds', dest='start_seconds', type=int, default=0, help='start time in secs')
    parser.add_argument('-e', '--end_seconds', dest='end_seconds', type=int, default=900, help='end time in secs')

    main(parser.parse_args())
