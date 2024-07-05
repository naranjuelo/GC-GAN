import cv2
import numpy as np
import h5py
import os
import mediapipe_aux
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


mapping = {
    'skin':   1,
    'eye':  2,
    'eyebrow': 3,
    'nose':  4,
    'mouth': 5,
}

save_file_faces = '/host/data_gan/ethx_segm2/faces_cor/'
save_file_faces_patch = '/host/data_gan/ethx_segm2/faces_rgb/'
save_file_eyes = '/host/data_gan/ethx_segm2/eyes/'
save_file_debug = '/host/data_gan/ethx_segm2/debug/'
save_file_eyes_rgb = '/host/data_gan/ethx_segm2/eyes_rgb/'
#img_file = 'face_sample.jpg'
#image = cv2.imread(img_file)

def get_mask(image, outname):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:

        h, w,_ = image.shape
        # add margin because preprocessed data are not perfectly aligned
        bigger_im = np.zeros([500, 500, 3], dtype=np.uint8)
        y0b = int((500-h)*0.5)
        x0b = int((500-w)*0.5)
        bigger_im[y0b:y0b+h, x0b:x0b+w] = image
        #cv2.imwrite('tmp.jpg',bigger_im)
        image = bigger_im
        h, w,_ = image.shape
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            mask_all = eyes_region = eyes_region_mask = eyes_region_rgb = None
            return mask_all, eyes_region, eyes_region_mask, eyes_region_rgb

        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            # print('face_landmarks:', face_landmarks)
            draw_mesh = False
            if draw_mesh:
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1))
            #cv2.imwrite('dummy_im_an.jpg', annotated_image)

            # Face meshed fitted, now crop eye regions:
            for face_landmarks in results.multi_face_landmarks:
                l_eye_landmarks, r_eye_landmarks, mouth_landmarks, l_eyebrow_landmarks, r_eyebrow_landmarks, nose_landmarks, all_landmarks = \
                    mediapipe_aux.__get_face_parts_all(
                        face_landmarks, w, h)

            # create hull array for convex hull points
            hullf = []
            maskf = np.zeros(image.shape, np.uint8)
            hullf.append(cv2.convexHull(np.asarray(all_landmarks), False))
            cv2.drawContours(maskf, hullf, 0, 1, 1, 8)
            cv2.fillPoly(maskf, [hullf[0]], (1,1,1))
            #cv2.imwrite('borrame.png', maskf)

            tmp_im = image.copy()
            for p in r_eye_landmarks:
                tmp_im = cv2.circle(tmp_im, (p[0],p[1]), 3, 200)
            for p in l_eye_landmarks:
                tmp_im = cv2.circle(tmp_im, (p[0],p[1]), 3, 200)
            for p in mouth_landmarks:
                tmp_im = cv2.circle(tmp_im, (p[0],p[1]), 3, 100)
            for p in r_eyebrow_landmarks:
                tmp_im = cv2.circle(tmp_im, (p[0],p[1]), 3, 50)
            for p in l_eyebrow_landmarks:
                tmp_im = cv2.circle(tmp_im, (p[0],p[1]), 3, 150)
            for p in nose_landmarks:
                tmp_im = cv2.circle(tmp_im, (p[0],p[1]), 3, 250)     

            mask_all = np.zeros(tmp_im.shape, np.float)
            parts= [r_eye_landmarks, l_eye_landmarks, mouth_landmarks, r_eyebrow_landmarks, l_eyebrow_landmarks, nose_landmarks]
            indexs = [2, 2, 5, 3, 3, 4]
            for i, part in enumerate(parts):
                hull = []
                mask = np.zeros(tmp_im.shape, np.uint8)
                hull = cv2.convexHull(np.array(part,dtype='int'))
                cv2.fillPoly(mask, [hull],(indexs[i], indexs[i], indexs[i]),cv2.LINE_8)
                ret,mask__ = cv2.threshold(mask,1,1,cv2.THRESH_BINARY)

                mask_i = np.zeros(tmp_im.shape, np.uint8)
                cv2.multiply(mask__,indexs[i],mask_i)
                #cv2.imwrite(save_file_debug + '00dummy'+str(i)+'.png', mask_i)
                mask_all += mask_i

                cv2.fillPoly(tmp_im, [hull],(i*10,255,0))
                #cv2.imwrite(save_file_debug + 'dummy'+str(i)+'.jpg', mask2)
           # cv2.imwrite(save_file_debug + outname, tmp_im)
            #cv2.imwrite(save_file_debug + 'dummy.jpg', mask_all)
            noparts = np.where((mask_all[:,:,0]==0) & (mask_all[:,:,1]==0) & (mask_all[:,:,2]==0))
            parts = np.where((mask_all[:,:,0]>0) & (mask_all[:,:,1]>0) & (mask_all[:,:,2]>0))
            skin = np.where(((maskf[:,:,0]==1) & (maskf[:,:,1]==1) & (maskf[:,:,2]==1)))
            mcpy = mask_all.copy()
            mcpy_neg = mask_all.copy()
            mask_all[skin] = (1,1,1)
            mask_all_ = mask_all +mcpy
            mask_all_[mcpy_neg>0] = mask_all_[mcpy_neg>0] -1
            mask_all = mask_all_

            # Get eye images
            l_eye_center = (
                int(0.5 * (l_eye_landmarks[1][0] + l_eye_landmarks[2][0]) + 0.5),
                int(0.5 * (l_eye_landmarks[1][1] + l_eye_landmarks[2][1]) + 0.5)
            )

            r_eye_center = (
                int(0.5 * (r_eye_landmarks[1][0] + r_eye_landmarks[2][0]) + 0.5),
                int(0.5 * (r_eye_landmarks[1][1] + r_eye_landmarks[2][1]) + 0.5)
            )
            l_eye_img, bb_leye = mediapipe_aux.__crops_eye_AFFNet(
                image, l_eye_landmarks[1], l_eye_landmarks[2], w, h)
            r_eye_img, bb_reye = mediapipe_aux.__crops_eye_AFFNet(
                image, r_eye_landmarks[2], r_eye_landmarks[1], w, h)
            
            csize = max((1/0.35)*l_eye_img.shape[0], (1/0.35)*r_eye_img.shape[0])
            face = []
            face, eyes_region, bb_face, bb_crop = mediapipe_aux.__crops_face_AFFNet(tmp_im, l_eye_center, r_eye_center, mouth_landmarks, csize, w, h)

            face, eyes_region_mask, bb_face, bb_crop = mediapipe_aux.__crops_face_AFFNet(mask_all, l_eye_center, r_eye_center, mouth_landmarks, csize, w, h)

            face, eyes_region_rgb, bb_face, bb_crop = mediapipe_aux.__crops_face_AFFNet(image, l_eye_center, r_eye_center, mouth_landmarks, csize, w, h)
            

            cropped_face_mask = mask_all[bb_crop[0][1]:bb_crop[1][1], bb_crop[0][0]:bb_crop[1][0]]
           #cv2.imwrite('tmpp.png',cropped_face_mask)

            return cropped_face_mask, eyes_region, eyes_region_mask, eyes_region_rgb
            
if __name__ == '__main__':
	datadir = '/host/ethx_db/train/'
	sub_folders = os.listdir(datadir)
	for ii, sub_id in enumerate(sub_folders):
		print(sub_id)
		if sub_id == 'subject0099.h5' or sub_id == '00010.h5' or sub_id == '00110.h5' or sub_id == '00126.h5' or sub_id == '00178.h5' or sub_id == '00190.h5':
			continue
		input_file = datadir + sub_id
		fid = h5py.File(input_file, 'r')
		img_size = 256
		num_data = fid["face_patch"].shape[0]   # get the total number of samples inside the h5 file
		print('num_data: ', num_data)

		img_show = np.zeros((img_size*3, img_size*6, 3), dtype=np.uint8)  # initial a empty image

		gaze = []
		num_i = 0
		cont_glob = 0
		while num_i < 100000:
			for cnt in range(5): # frames of same subj
				for num_r in range(0, 3):   # we show them in 3 rows
					for num_c in range(0, 6):   # we show them in 6 columns
						if num_i >= num_data:
							num_i = 100000
							break

						face_patch = fid['face_patch'][num_i, :]  # the face patch
						
						if 'face_gaze' in fid.keys():
							gaze = fid['face_gaze'][num_i, :]   # the normalized gaze direction with size of 2 dimensions as horizontal and vertical gaze directions.
						frame_index = fid['frame_index'][num_i, 0]  # the frame index
						cam_index = fid['cam_index'][num_i, 0]   # the camera index
						face_mat_norm = fid['face_mat_norm'][num_i, 0]   # the rotation matrix during data normalization
						face_head_pose = fid['face_head_pose'][num_i, 0]  # the normalized head pose with size of 2 dimensions horizontal and vertical head rotations.
						print('frame ind ' + str(frame_index) + '; cam ' + str(cam_index))

						face_patch = cv2.resize(face_patch, (img_size, img_size))
						if frame_index > 524:  # if the image is captured under low lighting conditions, we do histogram equalization
							num_i = num_i + 1                    
							if num_i >= num_data:
								break
							continue
							img_yuv = cv2.cvtColor(face_patch, cv2.COLOR_BGR2YUV)
							img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
							face_patch = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
				#		cropped_imgs = detect_crop_eyes_face(face_patch)      

				#		if len(cropped_imgs) == 0:
				#			num_i = num_i + 1                    
				#			if num_i >= num_data:
				#				break
				#			cont_glob += 1
				#			continue
						sg0 = "{:.4f}".format(gaze[0])
						sg1 = "{:.4f}".format(gaze[1])
						sid = sub_id.split('.')[0].split('subject')[1]
						if cam_index == 1 or cam_index == 2 or cam_index == 3 or cam_index == 8:  
							outname = str(cam_index)+'_s' + str(sid) + '_' + str(cont_glob) + '_eregion_' + sg0 + '_' + sg1 + '_c_' + str(cam_index)+'.png'
							face_region_mask, eyes_region_dbg, eyes_region_mask, eyes_region_rgb = get_mask(face_patch, outname)

							if face_region_mask is None:
								num_i = num_i + 1                    
								if num_i >= num_data:
									break
								cont_glob += 1
								continue

							r,g,b= cv2.split(face_region_mask)
							face_region_mask = r
							r,g,b= cv2.split(eyes_region_mask)
							eyes_region_mask = r

							#cv2.imwrite(save_file_eyes + str(cam_index)+'_s' + str(sid) + '_' + str(cont_glob) + '_eregion_' + sg0 + '_' + sg1 + '_c_' + str(cam_index)+'.png', eyes_region_mask)
							cv2.imwrite(save_file_faces + str(cam_index)+'_s' + str(sid) + '_' + str(cont_glob) + '_eregion_' + sg0 + '_' + sg1 + '_c_' + str(cam_index)+'.png', face_region_mask)
							#cv2.imwrite(save_file_faces_patch + str(cam_index)+'_s' + str(sid) + '_' + str(cont_glob) + '_eregion_' + sg0 + '_' + sg1 + '_c_' + str(cam_index)+'.png', face_patch)

							#cv2.imwrite(save_file_debug +str(cam_index)+'_s' + str(sid) + '_' + str(cont_glob) + '_eregion_' + sg0 + '_' + sg1 + '_c_' + str(cam_index)+'.png', eyes_region_dbg)
							#cv2.imwrite(save_file_eyes_rgb +str(cam_index)+'_s' + str(sid) + '_' + str(cont_glob) + '_eregion_' + sg0 + '_' + sg1 + '_c_' + str(cam_index)+'.png', eyes_region_rgb)

						cont_glob += 1

						#if 'face_gaze' in fid.keys():
						#	face_patch = draw_gaze(face_patch, gaze)  # draw gaze direction on the face patch image

						#img_show[img_size*num_r:img_size*(num_r+1), img_size*num_c:img_size*(num_c+1)] = face_patch
						num_i = num_i + 1                    
						if num_i >= num_data:
							break
            
    #         #cv2.imwrite('./xgaze_224/imgs/' + str(sub_id).zfill(4) + '_numi_' + str(num_i) + '_fri_' + str(frame_index) +'.jpg', img_show)
            
    #         #elif input_key == 106:  # j key to previous
    #         #   num_i = num_i - 18*2
    #         #  if num_i < 0:
    #             #    num_i = 0
    # #      num_i = num_i + 18
		fid.close()
    #     f.close()