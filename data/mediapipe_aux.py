import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import math
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int):
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and \
               (value < 1 or math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TDO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return [x_px, y_px]

def get_eye_mouth_landmarks(face_landmarks: landmark_pb2.NormalizedLandmarkList,
                              image_cols: int, image_rows: int):
    r_eye_landmarks = []
    l_eye_landmarks = []
    mouth_landmarks = []

    for idx, landmark in enumerate(face_landmarks.landmark):
        if ((landmark.HasField('visibility') and
             landmark.visibility < 0.5) or
                (landmark.HasField('presence') and
                 landmark.presence < 0.5)):
            if idx in (7, 33, 133, 144, 145, 153, 154, 155, 157, 158, \
                       159, 160, 161, 163, 173, 246, 469, 470, 471, 472):
                r_eye_landmarks.append((-1, -1))
            elif idx in (249, 263, 362, 373, 374, 380, 381, 382, 384, 385, \
                         386, 387, 388, 390, 398, 466, 474, 475, 476, 477):
                l_eye_landmarks.append((-1, -1))
            elif idx in (78, 308, 17, 0):
                mouth_landmarks.append((-1, -1))
            continue

        landmark_px = normalized_to_pixel_coordinates(
            normalized_x=landmark.x, normalized_y=landmark.y,
            image_width=image_cols, image_height=image_rows)
        if not landmark_px:
            continue

        if idx in (7, 33, 133, 144, 145, 153, 154, 155, 157, 158, \
                   159, 160, 161, 163, 173, 246, 469, 470, 471, 472):
            r_eye_landmarks.append(landmark_px)
        elif idx in (249, 263, 362, 373, 374, 380, 381, 382, 384, 385, \
                     386, 387, 388, 390, 398, 466, 474, 475, 476, 477):
            l_eye_landmarks.append(landmark_px)
        elif idx in (78, 308, 17, 0):
            mouth_landmarks.append(landmark_px)
    return l_eye_landmarks, r_eye_landmarks, mouth_landmarks

# mediapipe keypoint IDs for different face components
r_eye_ids = (246, 161, 160, 159, 158, 157, 173, 33, 7, 163, 144, 145, 153, 154, 155, 133)
l_eye_ids = (466, 388, 387, 386, 385, 384, 398,
  263, 249, 390, 373, 374, 380, 381, 382, 362)
r_iris_ids = (469, 470, 471, 472)
l_iris_ids = (474, 475, 476, 477)
r_eyebrow_ids = (70, 63, 105, 66, 107, 55, 46, 53, 52, 65)
l_eyebrow_ids = (300, 293, 334, 296, 336, 285,  276, 283, 282, 295)
mouth_ids = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291)
nose_ids = (1,2,98,327, 189, 417,64, 102, 331, 294)

def refine_iris_lnd(iris_lnd):
    new_iris_lnd = []
    (l_cx, l_cy), radius = cv2.minEnclosingCircle(iris_lnd)
    center = np.array([l_cx, l_cy], dtype=np.int32)
    cv2.circle(annotated_image, center, int(radius), (255, 0, 255), 1, cv2.LINE_AA)
    return new_iris_lnd

def get_face_parts_all(face_landmarks: landmark_pb2.NormalizedLandmarkList,
                              image_cols: int, image_rows: int):
    r_eye_landmarks = []
    l_eye_landmarks = []
    mouth_landmarks = []
    nose_landmarks = []
    r_eyebrow_landmarks = []
    l_eyebrow_landmarks = []
    r_iris_landmarks = []
    l_iris_landmarks = []
    r_iris = []
    l_iris = []
    all_landmarks = []

    for idx, landmark in enumerate(face_landmarks.landmark):
        if ((landmark.HasField('visibility') and
             landmark.visibility < 0.5) or
                (landmark.HasField('presence') and
                 landmark.presence < 0.5)):
            if idx in r_eye_ids:
                r_eye_landmarks.append((-1, -1))
            elif idx in l_eye_ids:
                l_eye_landmarks.append((-1, -1))
            elif idx in (78, 308, 17, 0):
                mouth_landmarks.append((-1, -1))
            elif idx in (469, 470, 471, 472):
                r_iris.append((-1,-1))
            elif idx in (474, 475, 476, 477):
                l_iris.append((-1,-1))
            continue

        landmark_px = normalized_to_pixel_coordinates(
            normalized_x=landmark.x, normalized_y=landmark.y,
            image_width=image_cols, image_height=image_rows)
        if not landmark_px:
            continue

        if idx in r_eye_ids:
            r_eye_landmarks.append(landmark_px)
           
        elif idx in l_eye_ids:
            l_eye_landmarks.append(landmark_px)
        elif idx in mouth_ids:
            mouth_landmarks.append(landmark_px)
        elif idx in nose_ids:
            nose_landmarks.append(landmark_px)
        elif idx in l_eyebrow_ids:
            l_eyebrow_landmarks.append(landmark_px)
        elif idx in r_eyebrow_ids:
            r_eyebrow_landmarks.append(landmark_px)
        elif idx in r_iris_ids:
            r_iris_landmarks.append(landmark_px)
        elif idx in l_iris_ids:
            l_iris_landmarks.append(landmark_px)
           
        all_landmarks.append(landmark_px)
        #r_iris = refine_iris_lnd(r_iris)
        #l_iris = refine_iris_lnd(l_iris)
    return l_eye_landmarks, r_eye_landmarks, mouth_landmarks, l_eyebrow_landmarks, r_eyebrow_landmarks, nose_landmarks, r_iris_landmarks,l_iris_landmarks, all_landmarks


def crop_eye_region(img, l_corner, r_corner, w, h):
    dX = abs(r_corner[0] - l_corner[0])

    desired_eye_x = 1.7*dX # we choose fixed size for the crop depending on eye coordinates
    eye_center_x = (r_corner[0] + l_corner[0])*0.5
    eye_center_y = (r_corner[1] + l_corner[1])*0.5
    l_corner_crop_x = max(0, int(eye_center_x - int(desired_eye_x*0.5)))
    r_corner_crop_x = min(w, int(eye_center_x + int(desired_eye_x*0.5)))
    l_corner_crop_y = max(0, int(eye_center_y - int(desired_eye_x*0.5)))
    r_corner_crop_y = min(h, int(eye_center_y + int(desired_eye_x*0.5)))

    cropped_img = img[l_corner_crop_y:r_corner_crop_y, l_corner_crop_x: r_corner_crop_x]

    return cropped_img, [(l_corner_crop_x, l_corner_crop_y), (r_corner_crop_x, r_corner_crop_y)]

def crop_face_regions(img, l_center, r_center, mouth_lnds, out_size, w, h):
        # We         set         the         face         center
        # as the mid point of the average eye coordinates and the average
        # mouth coordinates. Face bounding box size is 1/0.3 times eye
        # bounding box size. We use the same image resolution settings
        # as in the GazeCapture dataset.
        avg_eyes_y = int((r_center[1] + l_center[1])*0.5)
        avg_eyes_x = int((r_center[0] + l_center[0])*0.5)

        size_ = 250
        avg_mouth_x = avg_mouth_y = 0
        n_mtl = 0
        for ml in mouth_lnds:
            avg_mouth_x += ml[0]
            avg_mouth_y += ml[1]
            n_mtl += 1
        if 0 == n_mtl:
            avg_mouth_x = avg_eyes_x
            avg_mouth_y = avg_eyes_y 
        else:
            avg_mouth_x = avg_mouth_x/n_mtl
            avg_mouth_y = avg_mouth_y/n_mtl

        avg_face_x = int((avg_mouth_x + avg_eyes_x) * 0.5)
        avg_face_y = int((avg_mouth_y + avg_eyes_y) * 0.5)

        y0f = max(0, int((avg_face_y-out_size*0.5)))
        y1f = min(h, int((avg_face_y+out_size*0.5)))
        x0f = max(0, int((avg_face_x-out_size*0.5)))
        x1f = min(w, int((avg_face_x+out_size*0.5)))
        c = img.copy()
        cropped_img_face = c[y0f:y1f, x0f:x1f]        

        y0 = max(0, int((avg_face_y-out_size*0.5)))
        y1 = min(h, int(avg_face_y))
        xf0 = int((avg_face_x-out_size*0.5))
        xf1 = int((avg_face_x+out_size*0.5))
        x0 = max(0, xf0)
        x1 = min(w, xf1)
        c = img.copy()              
        # centered eyes (maybe some image is not visible because of their processing)
        cropped_img_face = cv2.resize(cropped_img_face, (size_,size_), interpolation=cv2.INTER_NEAREST)
        eyes_region = np.zeros([size_, size_, 3], dtype=np.uint8)

        eyes_region[int(size_*0.25):int(size_*0.75), 0:size_] = cropped_img_face[0:int(size_*0.5), 0:size_]

        return cropped_img_face, eyes_region, [(x0, y0), (x1, y1)], [(x0f, y0f), (x1f, y1f)]

# For static images:
IMAGE_FILES = ['tmp.jpg']
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

if __name__ == "__main__":
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:
      for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        w,h,_ = image.shape
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
          continue
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1))

        # Face meshed fitted, now crop eye regions:
        for face_landmarks in results.multi_face_landmarks:
            l_eye_landmarks, r_eye_landmarks, mouth_landmarks = \
                get_eye_mouth_landmarks(
                    face_landmarks, w, h)
        # Get eye images
        l_eye_center = (
            int(0.5 * (l_eye_landmarks[1][0] + l_eye_landmarks[2][0]) + 0.5),
            int(0.5 * (l_eye_landmarks[1][1] + l_eye_landmarks[2][1]) + 0.5)
        )

        r_eye_center = (
            int(0.5 * (r_eye_landmarks[1][0] + r_eye_landmarks[2][0]) + 0.5),
            int(0.5 * (r_eye_landmarks[1][1] + r_eye_landmarks[2][1]) + 0.5)
        )
        l_eye_img, bb_leye = crop_eye_region(
            image, l_eye_landmarks[1], l_eye_landmarks[2], w, h)
        r_eye_img, bb_reye = crop_eye_region(
            image, r_eye_landmarks[2], r_eye_landmarks[1], w, h)

        cv2.imwrite('annotated_image.png', annotated_image)
        cv2.imwrite('leye.png', l_eye_img)
        cv2.imwrite('reye.png', r_eye_img)