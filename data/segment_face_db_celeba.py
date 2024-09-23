import os
import numpy as np
import mediapipe_aux
import mediapipe as mp
import cv2
import argparse

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def draw_mask(image, maskf, r_eye_landmarks, l_eye_landmarks, mouth_landmarks, r_eyebrow_landmarks, l_eyebrow_landmarks, nose_landmarks, r_iris_landmarks, l_iris_landmarks):
    tmp_im = image.copy()
    for p in r_eye_landmarks:
        tmp_im = cv2.circle(tmp_im, (p[0], p[1]), 3, 200)
    for p in l_eye_landmarks:
        tmp_im = cv2.circle(tmp_im, (p[0], p[1]), 3, 200)
    for p in mouth_landmarks:
        tmp_im = cv2.circle(tmp_im, (p[0], p[1]), 3, 100)
    for p in r_eyebrow_landmarks:
        tmp_im = cv2.circle(tmp_im, (p[0], p[1]), 3, 50)
    for p in l_eyebrow_landmarks:
        tmp_im = cv2.circle(tmp_im, (p[0], p[1]), 3, 150)
    for p in nose_landmarks:
        tmp_im = cv2.circle(tmp_im, (p[0], p[1]), 3, 250)

    mask_all = np.zeros(tmp_im.shape, float)
    parts = [r_eye_landmarks, l_eye_landmarks, mouth_landmarks, r_eyebrow_landmarks, l_eyebrow_landmarks,
             nose_landmarks, r_iris_landmarks, l_iris_landmarks]
    indexs = [2, 2, 7, 3, 3, 4, 6, 6]
    for i, part in enumerate(parts):
        if i == (len(parts)-2):
            break
        hull = []
        mask = np.zeros(tmp_im.shape, np.uint8)
        hull = cv2.convexHull(np.array(part, dtype='int'))
        cv2.fillPoly(mask, [hull], (indexs[i],
                     indexs[i], indexs[i]), cv2.LINE_8)
        ret, mask__ = cv2.threshold(mask, 1, 1, cv2.THRESH_BINARY)
        mask_i = np.zeros(tmp_im.shape, np.uint8)
        cv2.multiply(mask__, indexs[i], mask_i)
        mask_all += mask_i
        cv2.fillPoly(tmp_im, [hull], (i*10, 255, 0))

    # draw iris
    mask_i = np.zeros(tmp_im.shape, np.uint8)
    (r_cx, r_cy), radius_r = cv2.minEnclosingCircle(np.array(r_iris_landmarks))
    center_r = np.array([r_cx, r_cy], dtype=np.int32)
    cv2.circle(mask_i, center_r, int(radius_r-1), (6, 6, 6), -1, cv2.FILLED)
    (l_cx, l_cy), radius_l = cv2.minEnclosingCircle(np.array(l_iris_landmarks))
    center_l = np.array([l_cx, l_cy], dtype=np.int32)
    cv2.circle(mask_i, center_l, int(radius_l-1), (6, 6, 6), -1, cv2.FILLED)
    mask_all += mask_i
    reskin = np.where((mask_all[:, :, 0] == 6) | (
        mask_all[:, :, 1] == 6) | (mask_all[:, :, 2] == 6))
    reiris = np.where((mask_all[:, :, 0] == 8) | (
        mask_all[:, :, 1] == 8) | (mask_all[:, :, 2] == 8))
    mask_all[reskin] = (1, 1, 1)
    mask_all[reiris] = (5, 5, 5)
    remouth = np.where((mask_all[:, :, 0] == 7) | (
        mask_all[:, :, 1] == 7) | (mask_all[:, :, 2] == 7))
    mask_all[remouth] = (6, 6, 6)

    #parts = np.where((mask_all[:, :, 0] > 0) & (
    #    mask_all[:, :, 1] > 0) & (mask_all[:, :, 2] > 0))
    skin = np.where(((maskf[:, :, 0] == 1) & (
        maskf[:, :, 1] == 1) & (maskf[:, :, 2] == 1)))
    mcpy = mask_all.copy()
    mcpy_neg = mask_all.copy()
    mask_all[skin] = (1, 1, 1)
    mask_all_ = mask_all + mcpy
    mask_all_[mcpy_neg > 0] = mask_all_[mcpy_neg > 0] - 1
    mask_all = mask_all_
    return mask_all, tmp_im

def get_mask(image):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        h = 512
        w = 512
        image = cv2.resize(image, (h,w))
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            mask_all = eyes_region = eyes_region_mask = eyes_region_rgb = frgb = fbb = None
            return mask_all, eyes_region, eyes_region_mask, eyes_region_rgb, frgb, fbb, fbb, fbb ,fbb

        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            draw_mesh = False
            if draw_mesh:
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1))

            # Face mesh fitted, crop eye regions:
            for face_landmarks in results.multi_face_landmarks:
                l_eye_landmarks, r_eye_landmarks, mouth_landmarks, l_eyebrow_landmarks, r_eyebrow_landmarks, nose_landmarks, r_iris_landmarks, l_iris_landmarks, all_landmarks = \
                    mediapipe_aux.get_face_parts_all(
                        face_landmarks, w, h)

            # create hull array for convex hull points
            hullf = []
            maskf = np.zeros(image.shape, np.uint8)
            hullf.append(cv2.convexHull(np.asarray(all_landmarks), False))
            cv2.drawContours(maskf, hullf, 0, 1, 1, 8)
            cv2.fillPoly(maskf, [hullf[0]], (1,1,1))
            cv2.imwrite('borrame.png', maskf)

            mask_all, _ = draw_mask(image, maskf, r_eye_landmarks,              l_eye_landmarks, mouth_landmarks,
                                         r_eyebrow_landmarks, l_eyebrow_landmarks, nose_landmarks, r_iris_landmarks, l_iris_landmarks)

            # Get eye images
            l_eye_center = (
                int(0.5 * (l_eye_landmarks[1][0] + l_eye_landmarks[2][0]) + 0.5),
                int(0.5 * (l_eye_landmarks[1][1] + l_eye_landmarks[2][1]) + 0.5)
            )

            r_eye_center = (
                int(0.5 * (r_eye_landmarks[1][0] + r_eye_landmarks[2][0]) + 0.5),
                int(0.5 * (r_eye_landmarks[1][1] + r_eye_landmarks[2][1]) + 0.5)
            )
            l_eye_img, _ = mediapipe_aux.crop_eye_region(
                image, l_eye_landmarks[1], l_eye_landmarks[2], w, h)
            r_eye_img, _ = mediapipe_aux.crop_eye_region(
                image, r_eye_landmarks[2], r_eye_landmarks[1], w, h)

            csize = max((1/0.35)*l_eye_img.shape[0], (1/0.35)*r_eye_img.shape[0])
            face = []
            # debug
            #face, eyes_region_deb, bb_face, bb_crop = mediapipe_aux.crop_face_regions(tmp_im, l_eye_center, r_eye_center, mouth_landmarks, csize, w, h)

            # mask
            face_mask, eyes_region_mask, bb_face, bb_crop = mediapipe_aux.crop_face_regions(mask_all, l_eye_center, r_eye_center, mouth_landmarks, csize, w, h)
            
            # rgb image
            face, eyes_region_rgb, bb_face, bb_crop = mediapipe_aux.crop_face_regions(image, l_eye_center, r_eye_center, mouth_landmarks, csize, w, h)

            #cropped_face_mask = mask_all[bb_crop[0][1]:bb_crop[1][1], bb_crop[0][0]:bb_crop[1][0]]

            return eyes_region_mask, eyes_region_rgb

def process_img(i, img_dataset_path, output_rgb, output_mask):
    img_output_path_eyes = output_rgb
    seg_trainset_path_eyes = output_mask
    
    # Read the image file
    input_path = os.path.join(img_dataset_path, f'{i}.jpg')
    face_patch = cv2.imread(input_path)
    
    # Process image
    eyes_region_mask, eyes_region_rgb = get_mask(face_patch)

    if eyes_region_mask is None:
        return 

    eyes_region_mask,_,_= cv2.split(eyes_region_mask)

    cv2.imwrite(os.path.join(seg_trainset_path_eyes, '0_s0_' + str(i) + '_eregion_0_0_c_0.png'), eyes_region_mask)
    cv2.imwrite(os.path.join(img_output_path_eyes, '0_s0_' + str(i) + '_eregion_0_0_c_0.png'), eyes_region_rgb)
    
    return

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a directory with CELEB data.")
    parser.add_argument('data_dir', type=str,
                        help='Directory where the data is located')
    parser.add_argument('save_dir_rgb', type=str,
                        help='Directory where the RGB images will be saved')
    parser.add_argument('save_dir_mask', type=str,
                        help='Directory where the mask images will be saved')
    args = parser.parse_args()
    
    datadir = args.data_dir
    output_rgb = args.save_dir_rgb
    output_mask = args.save_dir_mask
    create_directory(output_rgb)
    create_directory(output_mask)

    max_samples = 28000
    for i in range(0, max_samples):
        process_img(i, datadir, output_rgb, output_mask)