import numpy as np
import cv2

def write_avi(movie, file_name, frame_rate=30):
    """Save the Timeseries in a .avi movie file using OpenCV."""
    codec = cv2.FOURCC('I', 'Y', 'U', 'V') if hasattr(cv2, 'FOURCC') else cv2.VideoWriter_fourcc(*'IYUV')
    movie8 = movie.astype(np.uint8)
    vw = cv2.VideoWriter(file_name, codec, frame_rate, movie[0].shape[::-1], isColor=True)
    for frame in movie8:
        vw.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    vw.release()


def read_avi(file_name):
    """Loads Movie from a .avi video file."""
    cap = cv2.VideoCapture(file_name)
    use_cv2 = hasattr(cap, 'CAP_PROP_FRAME_COUNT')

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) if use_cv2 else cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) if use_cv2 else cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if use_cv2 else cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

    movie = np.zeros((length, height, width), dtype=np.uint8)
    for arr in movie:
        _, frame = cap.read()
        arr[:] = frame[:, :, 0]

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    return movie