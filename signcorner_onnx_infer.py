import json
import os.path

import numpy as np
import cv2
import onnxruntime

import warnings
warnings.filterwarnings("ignore")

class SignCorner_onnx_inference():
    def __init__(self, visFlg=False):
        # self.pose_checkpoint = '/media/zhangyi/pan2/PycharmProjects/Cloud/sign_corner_detection_v2.0/src/server/hrnet_128_128.onnx'
        # self.pose_checkpoint = '/media/zhangyi/pan2/PycharmProjects/mmpose/work_dirs/litehrnet_18_coco_128x128/litehrnet_onnxsim_128_128_210.onnx'
        self.pose_checkpoint = '/media/zhangyi/pan2/PycharmProjects/mmpose/work_dirs/shufflenetv2_18_coco_128x128/shufflenetv2_onnxsim_128_128_73.onnx'
        self.model_input_size = (128, 128)   #模型输入大小
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])
        self.padding = 1.25    #图片基于box的外扩倍数
        self.keypoints_num = 4
        self.visFlag = visFlg
        self.pre_post_process_meta = {}

    def box2sacle(self, bbox, scale):
        x, y, w, h = bbox[:4]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
        scale_w, scale_h = w*scale, h*scale
        if scale_w<scale_h:
            scale_w=scale_h
        else:
            scale_h=scale_w

        scale_x, scale_y = center[0]-scale_w*0.5, center[1]-scale_h*0.5
        if scale_x < 0:
            scale_x = 0
        if scale_y < 0:
            scale_y = 0
        self.pre_post_process_meta["scale_box"] = [scale_x, scale_y, scale_w, scale_h]
        return [scale_x, scale_y, scale_w, scale_h]

    def preprocess(self, img, det_result):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        box = det_result

        new_x, new_y, new_w, new_h = self.box2sacle(box, self.padding)
        image = image[int(new_y):int(new_y+new_h), int(new_x):int(new_x+new_w)]
        image = cv2.resize(image, self.model_input_size)
        crop_image = image.copy()

        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = image.transpose((2, 0, 1))  # (3, 96, 96)
        image = image[np.newaxis, :, :, :]  # (1, 3, 96, 96)

        image = np.array(image, dtype=np.float32)
        return image, crop_image

    def inference(self, model_input):
        session = onnxruntime.InferenceSession(self.pose_checkpoint)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        heatmap = session.run([output_name, session.get_outputs()[0].name], {input_name: model_input})
        return heatmap[0]

    def postprocess(self, heatmap, input_img, src_img, label):

        preds2input, maxvals = self.decode(heatmap)

        # Transform back to the image
        preds2image = self.transform_preds(preds2input)

        if label==1 or label==3:
            preds2image = self.delete_tri_point(preds2image)

        if self.visFlag:
            for i, keypoint in enumerate(preds2input[0]):
                x = int(keypoint[0])
                y = int(keypoint[1])
                cv2.circle(input_img, (x, y), radius=1, color=(0, 255, 0), thickness=4)
            cv2.imwrite(f'./data/vis_result_{self.model_input_size[0]}_{self.model_input_size[1]}.jpg', input_img)

            for i, keypoint in enumerate(preds2image[0]):
                x = int(keypoint[0])
                y = int(keypoint[1])
                cv2.circle(src_img, (x, y), radius=1, color=(0, 255, 0), thickness=4)
            cv2.imwrite(f'./data/vis_result_{src_img.shape[1]}_{src_img.shape[0]}.jpg', src_img)

        return preds2input[0], preds2image[0]

    def delete_tri_point(self, preds):
        """
        Note: category: 1: 三角形标牌
                        2: 四边形标牌
                        3: 三角形标牌背面
                        4: 四边形标牌背面
                        5: 圆形标牌
                        6: 圆形标牌背面
        Args:
            preds: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

        Returns: list

        """
        from scipy.spatial.distance import pdist

        pred_points = preds[0]
        assert len(pred_points) == 4

        deletFlag = 0
        for j in range(0, 2):
            for i in range(j + 1, len(pred_points) - 1):
                XY = np.vstack([pred_points[j], pred_points[i]])  # [[x1,x2],[y1,y2]]
                if pdist(XY) < 2:
                    deletFlag = 1
                    deletesigncorner = np.delete(pred_points, j, axis=0)  # 距离小于2的删除重复的一个就行
                    delet_result = deletesigncorner.tolist()
                    continue
            if deletFlag == 1:
                continue
        if deletFlag == 0:
            deletesigncorner = np.delete(pred_points, 1, axis=0)  # 没有距离小于２的删除第二个
            delet_result = deletesigncorner.tolist()

        return np.array([delet_result])

    def _get_max_preds(self, heatmaps):
        """Get keypoint predictions from score maps.

        Note:
            batch_size: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

        Returns:
            tuple: A tuple containing aggregated results.

            - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
            - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
        """
        assert isinstance(heatmaps,
                          np.ndarray), ('heatmaps should be numpy.ndarray')
        assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        N, K, _, W = heatmaps.shape
        heatmaps_reshaped = heatmaps.reshape((N, K, -1))
        idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
        maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
        preds[:, :, 0] = preds[:, :, 0] % W
        preds[:, :, 1] = preds[:, :, 1] // W

        preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
        return preds, maxvals

    def transform_preds(self, coords):
        """Get final keypoint predictions from heatmaps and apply scaling and
        translation to map them back to the image.

        Note:
            num_keypoints: K

        Args:
            coords (np.ndarray[K, ndims]):

                * If ndims=2, corrds are predicted keypoint location.
                * If ndims=4, corrds are composed of (x, y, scores, tags)
                * If ndims=5, corrds are composed of (x, y, scores, tags,
                  flipped_tags)

        Returns:
            np.ndarray: Predicted coordinates in the images.
        """
        assert coords.shape[1] in (2, 4, 5)

        scale_box = self.pre_post_process_meta["scale_box"]
        scale_x, scale_y = scale_box[2]/self.model_input_size[1], scale_box[3]/self.model_input_size[1]

        target_coords = np.copy(coords)
        target_coords[:, :, 0] = coords[:, :, 0] * scale_x + scale_box[0]
        target_coords[:, :, 1] = coords[:, :, 1] * scale_y + scale_box[1]

        return target_coords

    def decode(self, heatmaps):
        N, K, H, W = heatmaps.shape
        preds, maxvals = self._get_max_preds(heatmaps)

        for n in range(N):
            for k in range(K):
                heatmap = heatmaps[n][k]
                px = int(preds[n][k][0])
                py = int(preds[n][k][1])
                if 1 < px < W - 1 and 1 < py < H - 1:
                    diff = np.array([
                        heatmap[py][px + 1] - heatmap[py][px - 1],
                        heatmap[py + 1][px] - heatmap[py - 1][px]
                    ])
                    preds[n][k] += np.sign(diff) * .25

            # Transform back to the image
            for i in range(N):
                preds[i] = preds[i] * 4

        return preds, maxvals

def single_img_test():
    # img_path = './affline_img.jpg'

    # img_path = './data/test/38-1-190708103452970-001292.jpg'
    # box = [1605, 779, 192, 181]

    img_path = './data/test/test.jpg'
    box = [809, 258, 52, 59]
    src_img = cv2.imread(img_path)

    infer_enigne = SignCorner_onnx_inference(visFlg=False)
    input_data, crop_image = infer_enigne.preprocess(src_img, box)

    # crop_image = src_img.copy()
    # input_data = infer_enigne.test_preprocess(src_img, 0)

    heatmap = infer_enigne.inference(input_data)
    preds2model_input, preds2src_img = infer_enigne.postprocess(heatmap, crop_image, src_img, '01')

def parase_det_result():
    det_json_path = "./data/signcorner/annotations/val.json"
    image_dir = "./data/signcorner/val"
    result_save_dir = "./data/signcorner/val_test"

    val_data = json.load(open(det_json_path, 'r'))
    image_info = val_data["images"]
    annotations_info = val_data["annotations"]

    infer_enigne = SignCorner_onnx_inference(visFlg=False)
    img_obj_id, img_id = 0, 0
    for i, annotation in enumerate(annotations_info):
        # annotation = annotations_info[701]

        image_id = annotation["image_id"]

        if image_id != img_id:
            img_id = 0
        bbox = annotation["bbox"]
        category_id = annotation["category_id"]

        imgfile = image_info[image_id]["file_name"]
        print(f"INFO: Process [{i}]/[{len(annotations_info)}] imgfile: {imgfile}")
        imgpath = os.path.join(image_dir, imgfile)
        src_img =  cv2.imread(imgpath)
        input_data, crop_image = infer_enigne.preprocess(src_img, bbox)
        heatmap = infer_enigne.inference(input_data)
        preds2model_input, preds2src_img = infer_enigne.postprocess(heatmap, crop_image, src_img, category_id)

        img_name = imgfile.split(".")[0]
        crop_image = cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR)
        for i, keypoint in enumerate(preds2model_input):
            x = int(keypoint[0])
            y = int(keypoint[1])
            cv2.circle(crop_image, (x, y), radius=1, color=(0, 0, 255), thickness=4)
        vis_crop_img_save_path = os.path.join(result_save_dir, "128", f'{img_name}_{img_obj_id}.jpg')
        cv2.imwrite(vis_crop_img_save_path, crop_image)

        # for i, keypoint in enumerate(preds2src_img):
        #     x = int(keypoint[0])
        #     y = int(keypoint[1])
        #     cv2.circle(src_img, (x, y), radius=1, color=(0, 0, 255), thickness=4)
        # vis_src_img_save_path = os.path.join(result_save_dir, "src", f'{img_name}_{img_obj_id}.jpg')
        # cv2.imwrite(vis_src_img_save_path, src_img)

        if image_id == img_id:
            img_obj_id += 1
        else:
            img_obj_id = 0
            img_id = image_id



if __name__ == "__main__":
    # single_img_test()
    parase_det_result()


