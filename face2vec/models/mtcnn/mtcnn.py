"""
Code for MTCNN model taken and adapted from
[facenet-pytorch](https://github.com/timesler/facenet-pytorch/blob/master/models/mtcnn.py)

All credit goes to the original authors.
"""
from typing import Union, List

import torch
from PIL import Image
from torch import nn
import numpy as np
import os

from torchvision.transforms import transforms

from face2vec.models.mtcnn import PNet, RNet, ONet
from face2vec.models.mtcnn.utils import detect_face, extract_face, fixed_image_standardization


class MTCNN(nn.Module):
    """mtcnn face detection module.

    This class loads pretrained P-, R-, and O-nets and returns images cropped to include the face
    only, given raw input images of one of the following types:
        - PIL image or list of PIL images
        - numpy.ndarray (uint8) representing either a single image (3D) or a batch of images (4D).
    Cropped faces can optionally be saved to file
    also.
    
    Keyword Arguments:
        image_size {int} -- Output image size in pixels. The image will be square. (default: {160})
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image. 
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size (this is a bug in davidsandberg/facenet).
            (default: {0})
        min_face_size {int} -- Minimum face size to search for. (default: {20})
        thresholds {list} -- mtcnn face detection thresholds (default: {[0.6, 0.7, 0.7]})
        factor {float} -- Factor used to create a scaling pyramid of face sizes. (default: {0.709})
        post_process {bool} -- Whether to post process images tensors before returning.
            (default: {True})
        select_largest {bool} -- If True, if multiple faces are detected, the largest is returned.
            If False, the face with the highest detection probability is returned.
            (default: {True})
        selection_method {string} -- Which heuristic to use for selection. Default None. If
            specified, will override select_largest:
                    "probability": highest probability selected
                    "largest": largest box selected
                    "largest_over_threshold": largest box over a certain probability selected
                    "center_weighted_size": box size minus weighted squared offset from image center
                (default: {None})
        keep_all {bool} -- If True, all detected faces are returned, in the order dictated by the
            select_largest parameter. If a save_path is specified, the first face is saved to that
            path and the remaining faces are saved to <save_path>1, <save_path>2 etc.
            (default: {False})
        device {torch.device} -- The device on which to run neural net passes. Image tensors and
            models are copied to this device before running forward passes. (default: {None})
    """

    def __init__(
            self, image_size=160, margin=0, min_face_size=20,
            thresholds=None, factor=0.709, post_process=True,
            select_largest=True, selection_method=None, keep_all=False, device=None
    ):
        super().__init__()

        if thresholds is None:
            thresholds = [0.6, 0.7, 0.7]

        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.post_process = post_process
        self.select_largest = select_largest
        self.keep_all = keep_all
        self.selection_method = selection_method

        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.to(device)

        if not self.selection_method:
            self.selection_method = 'largest' if self.select_largest else 'probability'

    def _convert_img(self, img: Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]):
        if isinstance(img, Image.Image):
            img = transforms.PILToTensor()(img).unsqueeze(0).to(self.device)
        elif isinstance(img, list):
            img = torch.stack([transforms.PILToTensor()(i) for i in img]).to(self.device)
        elif isinstance(img, np.ndarray):
            if len(img.shape) <= 2:
                raise ValueError("Image must have 3 color channels")
            elif len(img.shape) == 3:
                img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            elif len(img.shape) == 4:
                img = torch.tensor(img).permute(0, 3, 1, 2).to(self.device)
        elif isinstance(img, torch.Tensor):
            if len(img.shape) == 3:
                img = img.unsqueeze(0)

            img = img.to(self.device)

        return img

    def forward(self, img: Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray], save_path=None, return_prob=False):
        """Run mtcnn face detection on a PIL image or numpy array. This method performs both
        detection and extraction of faces, returning tensors representing detected faces rather
        than the bounding boxes. To access bounding boxes, see the mtcnn.detect() method below.
        
        Arguments:
            img {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, torch.Tensor.
        
        Keyword Arguments:
            save_path {str} -- An optional save path for the cropped image. Note that when
                self.post_process=True, although the returned tensor is post processed, the saved
                face image is not, so it is a true representation of the face in the input image.
                If `img` is a list of images, `save_path` should be a list of equal length.
                (default: {None})
            return_prob {bool} -- Whether to return the detection probability.
                (default: {False})
        
        Returns:
            Union[torch.Tensor, tuple(torch.tensor, float)] -- If detected, cropped image of a face
                with dimensions 3 x image_size x image_size. Optionally, the probability that a
                face was detected. If self.keep_all is True, n detected faces are returned in an
                n x 3 x image_size x image_size tensor with an optional list of detection
                probabilities. If `img` is a list of images, the item(s) returned have an extra 
                dimension (batch) as the first dimension.

        Example:
        >>> from face2vec.models.mtcnn import mtcnn
        >>> mtcnn = mtcnn()
        >>> face_tensor, prob = mtcnn(img, save_path='face.png', return_prob=True)
        """

        img = self._convert_img(img)

        # Detect faces
        batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)

        # Select faces
        if not self.keep_all:
            batch_boxes, batch_probs, batch_points = self.select_boxes(
                batch_boxes, batch_probs, batch_points, img, method=self.selection_method
            )
        # Extract faces
        faces = self.extract(img, batch_boxes, save_path)

        if faces is None:
            return None

        if img.shape[0] == 1:
            faces = faces[0]
            batch_probs = batch_probs[0]

        if return_prob:
            return faces, batch_probs
        else:
            return faces

    def detect(self, img, landmarks=False):
        """Detect all faces in PIL image and return bounding boxes and optional facial landmarks.

        This method is used by the forward method and is also useful for face detection tasks
        that require lower-level handling of bounding boxes and facial landmarks (e.g., face
        tracking). The functionality of the forward function can be emulated by using this method
        followed by the extract_face() function.
        
        Arguments:
            img {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, torch.Tensor, or list.

        Keyword Arguments:
            landmarks {bool} -- Whether to return facial landmarks in addition to bounding boxes.
                (default: {False})
        
        Returns:
            tuple(numpy.ndarray, list) -- For N detected faces, a tuple containing an
                Nx4 array of bounding boxes and a length N list of detection probabilities.
                Returned boxes will be sorted in descending order by detection probability if
                self.select_largest=False, otherwise the largest face will be returned first.
                If `img` is a list of images, the items returned have an extra dimension
                (batch) as the first dimension. Optionally, a third item, the facial landmarks,
                are returned if `landmarks=True`.

        Example:
        >>> from face2vec.models.mtcnn import mtcnn        >>> from PIL import Image, ImageDraw
        >>> mtcnn = mtcnn(keep_all=True)
        >>> boxes, probs, points = mtcnn.detect(img, landmarks=True)
        >>> # Draw boxes and save faces
        >>> img_draw = img.copy()
        >>> draw = ImageDraw.Draw(img_draw)
        >>> for i, (box, point) in enumerate(zip(boxes, points)):
        ...     draw.rectangle(box.tolist(), width=5)
        ...     for p in point:
        ...         draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
        ...     extract_face(img, box, save_path='detected_face_{}.png'.format(i))
        >>> img_draw.save('annotated_faces.png')
        """

        img = self._convert_img(img)

        with torch.no_grad():
            batch_boxes, batch_points = detect_face(
                img, self.min_face_size,
                self.pnet, self.rnet, self.onet,
                self.thresholds, self.factor,
                self.device
            )

        boxes, probs, points = [], [], []
        for box, point in zip(batch_boxes, batch_points):
            box = np.array(box)
            point = np.array(point)
            if len(box) == 0:
                boxes.append(None)
                probs.append([None])
                points.append(None)
            elif self.select_largest:
                box_order = np.argsort((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]))[::-1]
                box = box[box_order]
                point = point[box_order]
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
                points.append(point)
            else:
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
                points.append(point)
        boxes = np.array(boxes, dtype=object)
        probs = np.array(probs, dtype=object)
        points = np.array(points, dtype=object)

        if landmarks:
            return boxes, probs, points

        return boxes, probs

    def select_boxes(
            self, all_boxes, all_probs, all_points, imgs, method='probability', threshold=0.9,
            center_weight=2.0
    ):
        """Selects a single box from multiple for a given image using one of multiple heuristics.

        Arguments:
                all_boxes {np.ndarray} -- Ix0 ndarray where each element is a Nx4 ndarry of
                    bounding boxes for N detected faces in I images (output from self.detect).
                all_probs {np.ndarray} -- Ix0 ndarray where each element is a Nx0 ndarry of
                    probabilities for N detected faces in I images (output from self.detect).
                all_points {np.ndarray} -- Ix0 ndarray where each element is a Nx5x2 array of
                    points for N detected faces. (output from self.detect).
                imgs {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, torch.Tensor, or list.

        Keyword Arguments:
                method {str} -- Which heuristic to use for selection:
                    "probability": highest probability selected
                    "largest": largest box selected
                    "largest_over_theshold": largest box over a certain probability selected
                    "center_weighted_size": box size minus weighted squared offset from image center
                    (default: {'probability'})
                threshold {float} -- theshold for "largest_over_threshold" method. (default: {0.9})
                center_weight {float} -- weight for squared offset in center weighted size method.
                    (default: {2.0})

        Returns:
                tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray) -- nx4 ndarray of bounding boxes
                    for n images. Ix0 array of probabilities for each box, array of landmark points.
        """
        selected_boxes, selected_probs, selected_points = [], [], []
        for boxes, points, probs, img in zip(all_boxes, all_points, all_probs, imgs):

            if boxes is None:
                continue

            # If at least 1 box found
            boxes = np.array(boxes)
            probs = np.array(probs)
            points = np.array(points)
            box_order = None

            if method == 'largest':
                box_order = np.argsort((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[::-1]
            elif method == 'probability':
                box_order = np.argsort(probs)[::-1]
            elif method == 'center_weighted_size':
                box_sizes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                img_center = (img.width / 2, img.height / 2)
                box_centers = np.array(list(zip((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2)))
                offsets = box_centers - img_center
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 1)
                box_order = np.argsort(box_sizes - offset_dist_squared * center_weight)[::-1]
            elif method == 'largest_over_threshold':
                box_mask = probs > threshold
                boxes = boxes[box_mask]
                box_order = np.argsort((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[::-1]
                if sum(box_mask) == 0:
                    selected_boxes.append(None)
                    selected_probs.append([None])
                    selected_points.append(None)
                    continue

            if box_order is None:
                raise Exception("Invalid selection method: {}".format(method))

            box = boxes[box_order][[0]]
            prob = probs[box_order][[0]]
            point = points[box_order][[0]]
            selected_boxes.append(box)
            selected_probs.append(prob)
            selected_points.append(point)

        return np.array(selected_boxes), np.array(selected_probs), np.array(selected_points)

    def extract(self, img, batch_boxes, save_path):
        # Parse save path(s)
        if save_path is not None:
            if isinstance(save_path, str):
                save_path = [save_path]
        else:
            save_path = [None for _ in range(len(img))]

        # Process all bounding boxes
        faces = []
        for im, box_im, path_im in zip(img, batch_boxes, save_path):
            if box_im is None:
                faces.append(None)
                continue

            if not self.keep_all:
                box_im = box_im[[0]]

            faces_im = []
            for i, box in enumerate(box_im):
                face_path = path_im
                if path_im is not None and i > 0:
                    save_name, ext = os.path.splitext(path_im)
                    face_path = save_name + '_' + str(i + 1) + ext

                face = extract_face(im, box, self.image_size, self.margin, face_path)
                if self.post_process:
                    face = fixed_image_standardization(face)
                faces_im.append(face)

            if self.keep_all:
                faces_im = torch.stack(faces_im)
            else:
                faces_im = faces_im[0]

            faces.append(faces_im)

        if len(faces) == 0:
            return None

        return torch.stack(faces)
