from abc import abstractmethod
import os
import random
import time

from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms

from yolov7_utils import LoadImages, attempt_load, check_img_size, non_max_suppression, plot_one_box, scale_coords, select_device, time_synchronized


class generic_segmenter:
    def __init__(self, path_to_model: str) -> None:
        """Method to read model from file"""
        print(f'Readinf model from {path_to_model}')
        self.model_path = path_to_model

    @abstractmethod
    def model_info(self) -> None:
        """Method to show info and metadata"""
        pass

    @abstractmethod
    def segment(self, image):
        pass


class dummy_segmenter(generic_segmenter):
    """Dummy segmenter.

    It is a simple example what your segmenter need to work.
    """
    def __init__(self, path_to_model: str) -> None:
        """Method to read model from file"""
        print(f'Reading model from {path_to_model}')
        self.model_path = path_to_model

    def model_info(self) -> None:
        """Method to show info and metadata"""
        print(f'Model: {self.model_path}')
    
    def segment(self, image):
        """Method to do segmenting magic

        On input may be cv::Mat, torch::Tensor or PIL::Image.
        """
        print(f'Segmenting...\nEnded in: 0 seconds.')
        return image


class YoloV7_segmenter(generic_segmenter):
    """YoloV7 segmenter.

    It is segmenter for making inference with models from Yolov7.
    """
    def __init__(self, path_to_model: str) -> None:
        """Method to read model from file"""
        print(f'Reading model from {path_to_model}')
        self.device = select_device('')

        self.model_path = path_to_model

        # self.model = torch.jit.load(path_to_model)
        self.model = attempt_load(self.model_path, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.half = self.device.type != 'cpu'
        if self.half:
            self.model.half()
        
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        # self.model.eval()

    def model_info(self) -> None:
        """Method to show info and metadata"""
        print(f'Model: {self.model}')
    
    def segment(self, image):
        """Method to do segmenting magic

        On input may be cv::Mat, torch::Tensor or PIL::Image.
        """
        self.imgsz = check_img_size(640, s=self.stride)
        # convert PIL image to opencv:
        pil_image = image.convert('RGB')
        open_cv_image = np.array(pil_image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        dataset = LoadImages(open_cv_image, img_size=self.imgsz, stride=self.stride)

        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=True)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if True:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):

                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)

        # Print time (inference + NMS)
        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(im0)
        return im_pil


class DeepLabV3_segmenter(generic_segmenter):
    def __init__(self, path_to_model: str) -> None:
        super().__init__(path_to_model)
        self.model = torch.load(self.model_path)
        self.model.eval()

    def model_info(self) -> None:
        print(self.model)

    def segment(self, image: Image) -> Image:
        input_image = image.convert('RGB')
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype('uint8')
        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
        r.putpalette(colors)
        return r


def get_segmenter(segmenter_name: str, model_path: str) -> generic_segmenter:
    """Method for create segmenter (pseudo fabric).
    
    Ready segmenters are:
    - DUMMY_SEGMENTER
    - DeepLabV3
    WIP:
    - YoloV7
    """

    print(segmenter_name, model_path)

    if segmenter_name == 'DUMMY_SEGMENTER':
        return dummy_segmenter(r'C:\system32\rule34.pt')
    elif segmenter_name == 'YoloV7':
        print('YoloV7')
        return YoloV7_segmenter(r'./ml_models/YoloV7.pt')
    elif segmenter_name == 'droniada':
        print('YoloV7')
        return YoloV7_segmenter(model_path)
    elif segmenter_name == 'DeepLabV3':
        if os.path.isfile(model_path) and model_path.split('.')[-1] in ['pt', 'pth']:
            return DeepLabV3_segmenter(model_path)
        else:
            raise RuntimeError(f'Path {model_path} doesnt exist :/. Did you download models? Did you even read README.md bi$ch?')
    else:
        raise RuntimeError(f'Model {segmenter_name} is not supported.')
