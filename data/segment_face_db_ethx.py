import cv2
import numpy as np
import h5py
import os
import mediapipe_aux
import mediapipe as mp
import argparse

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")


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

    parts = np.where((mask_all[:, :, 0] > 0) & (
        mask_all[:, :, 1] > 0) & (mask_all[:, :, 2] > 0))
    skin = np.where(((maskf[:, :, 0] == 1) & (
        maskf[:, :, 1] == 1) & (maskf[:, :, 2] == 1)))
    mcpy = mask_all.copy()
    mcpy_neg = mask_all.copy()
    mask_all[skin] = (1, 1, 1)
    mask_all_ = mask_all + mcpy
    mask_all_[mcpy_neg > 0] = mask_all_[mcpy_neg > 0] - 1
    mask_all = mask_all_
    return mask_all, tmp_im


def get_mask(image_raw):
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:

        h, w, _ = image_raw.shape
        # add margin because preprocessed data are not perfectly aligned
        bigger_im = np.zeros([500, 500, 3], dtype=np.uint8)
        y0b = int((500-h)*0.5)
        x0b = int((500-w)*0.5)
        bigger_im[y0b:y0b+h, x0b:x0b+w] = image_raw
        image = bigger_im
        h, w, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            mask_all = eyes_region = eyes_region_mask = eyes_region_rgb = None
            return mask_all, eyes_region, eyes_region_mask, eyes_region_rgb, None

        for face_landmarks in results.multi_face_landmarks:
            # Face meshed fitted, now draw different component regions for mask:
            for face_landmarks in results.multi_face_landmarks:
                l_eye_landmarks, r_eye_landmarks, mouth_landmarks, l_eyebrow_landmarks, r_eyebrow_landmarks, nose_landmarks, r_iris_landmarks, \
                    l_iris_landmarks, all_landmarks \
                    = mediapipe_aux.get_face_parts_all(
                        face_landmarks, w, h)

            # create hull array for convex hull points
            hullf = []
            maskf = np.zeros(image.shape, np.uint8)
            hullf.append(cv2.convexHull(np.asarray(all_landmarks), False))
            cv2.drawContours(maskf, hullf, 0, 1, 1, 8)
            cv2.fillPoly(maskf, [hullf[0]], (1, 1, 1))

            mask_all, tmp_im = draw_mask(image, maskf, r_eye_landmarks, l_eye_landmarks, mouth_landmarks,
                                         r_eyebrow_landmarks, l_eyebrow_landmarks, nose_landmarks, r_iris_landmarks, l_iris_landmarks)

            # Get eye images
            l_eye_center = (
                int(0.5 * (l_eye_landmarks[1][0] +
                    l_eye_landmarks[2][0]) + 0.5),
                int(0.5 * (l_eye_landmarks[1][1] +
                    l_eye_landmarks[2][1]) + 0.5)
            )
            r_eye_center = (
                int(0.5 * (r_eye_landmarks[1][0] +
                    r_eye_landmarks[2][0]) + 0.5),
                int(0.5 * (r_eye_landmarks[1][1] +
                    r_eye_landmarks[2][1]) + 0.5)
            )
            l_eye_img, _ = mediapipe_aux.crop_eye_region(
                image, l_eye_landmarks[1], l_eye_landmarks[2], w, h)
            r_eye_img, _ = mediapipe_aux.crop_eye_region(
                image, r_eye_landmarks[2], r_eye_landmarks[1], w, h)
            csize = max(
                (1/0.35)*l_eye_img.shape[0], (1/0.35)*r_eye_img.shape[0])

            face = []
            # debug
            face, eyes_region, bb_face, bb_crop = mediapipe_aux.crop_face_regions(
                tmp_im, l_eye_center, r_eye_center, mouth_landmarks, csize, w, h)

            # mask
            face, eyes_region_mask, _, _ = mediapipe_aux.crop_face_regions(
                mask_all, l_eye_center, r_eye_center, mouth_landmarks, csize, w, h)

            # rgb image
            face, eyes_region_rgb, _, _ = mediapipe_aux.crop_face_regions(
                image, l_eye_center, r_eye_center, mouth_landmarks, csize, w, h)

            h, w, _ = image_raw.shape
            # remove originally added margin for face
            y0b = int((500 - h) * 0.5)
            x0b = int((500 - w) * 0.5)
            face_mask = mask_all[y0b:y0b + h, x0b:x0b + w]

            return face_mask, face, eyes_region, eyes_region_mask, eyes_region_rgb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some directories.")
    parser.add_argument('data_dir', type=str,
                        help='Directory where the data is located')
    parser.add_argument('save_dir_rgb', type=str,
                        help='Directory where the RGB images will be saved')
    parser.add_argument('save_dir_mask', type=str,
                        help='Directory where the mask images will be saved')
    args = parser.parse_args()

    img_size = 256
    datadir = args.data_dir
    create_directory(args.data_dir)
    create_directory(args.save_dir_rgb)
    create_directory(args.save_dir_mask)
    subj_files = os.listdir(datadir)

    for sub_id in subj_files:
        print(sub_id)
        # these files were problematic
        if sub_id == 'subject0099.h5' or sub_id == '00010.h5' or sub_id == '00110.h5' or sub_id == '00126.h5' or sub_id == '00178.h5' or sub_id == '00190.h5':
            continue
        input_file = datadir + sub_id
        fid = h5py.File(input_file, 'r')
        # get the total number of samples inside the h5 file
        num_data = fid["face_patch"].shape[0]

        gaze = []
        num_i = 0
        cont_glob = 0
        max_samples = 100000
        while num_i < max_samples:
            for cnt in range(5):  # frames of same subj
                if num_i >= num_data:
                    num_i = max_samples
                    break

                face_patch = fid['face_patch'][num_i, :]  # face patch
                if 'face_gaze' in fid.keys():
                    # the normalized gaze direction with size of 2 dimensions as horizontal and vertical gaze directions.
                    gaze = fid['face_gaze'][num_i, :]
                frame_index = fid['frame_index'][num_i, 0]  # the frame index
                cam_index = fid['cam_index'][num_i, 0]   # the camera index
                print('Frame ind ' + str(frame_index) +
                      '; cam ' + str(cam_index))

                face_patch = cv2.resize(face_patch, (img_size, img_size))
                # if the image is captured under low lighting conditions, we do histogram equalization (from ETHX dataset)
                if frame_index > 524:
                    num_i = num_i + 1
                    if num_i >= num_data:
                        break
                    continue  # we are not considering these images
                    # img_yuv = cv2.cvtColor(face_patch, cv2.COLOR_BGR2YUV)
                    # img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                    # face_patch = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

                sg0 = "{:.4f}".format(gaze[0])
                sg1 = "{:.4f}".format(gaze[1])
                sid = sub_id.split('.')[0].split('subject')[1]
                # camera IDs we use (frontal ones)
                if cam_index == 1 or cam_index == 2 or cam_index == 3 or cam_index == 8:
                    face_region_mask, face_rgb, eyes_region_dbg, eyes_region_mask, eyes_region_rgb = get_mask(
                        face_patch)

                    if face_region_mask is None:
                        num_i = num_i + 1
                        if num_i >= num_data:
                            break
                        cont_glob += 1
                        continue

                    face_region_mask = cv2.split(face_region_mask)[0]
                    eyes_region_mask = cv2.split(eyes_region_mask)[0]

                    fname = str(cam_index)+'_s' + str(sid) + '_' + str(cont_glob) + \
                        '_eregion_' + sg0 + '_' + sg1 + \
                        '_c_' + str(cam_index)+'.png'

                    cv2.imwrite(os.path.join(
                        args.save_dir_mask, fname), eyes_region_mask)
                    # cv2.imwrite(args.save_file_faces + fname, face_region_mask)
                    # cv2.imwrite(save_file_debug + fname, eyes_region_dbg)
                    cv2.imwrite(os.path.join(
                        args.save_dir_rgb, fname), eyes_region_rgb)

                    cont_glob += 1

                num_i = num_i + 1
                if num_i >= num_data:
                    break
        fid.close()
