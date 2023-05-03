from abc import abstractmethod
import os

from PIL import Image
import torch
from torchvision import transforms


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


class DeepLabV3_segmenter(generic_segmenter):
    def __init__(self, path_to_model: str) -> None:
        super().__init__(path_to_model)
        self.model = torch.load(self.model_path)
        self.model.eval()

    def model_info(self) -> None:
        print(self.model)
    
    def segment(self, image: Image) -> Image:
        input_image = image.convert("RGB")
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
        colors = (colors % 255).numpy().astype("uint8")
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
        print('NOT YET IMPLEMENTED, USING DUMMY_SEGMENTER (YES IT DOES NOTHING RIGHT NOW.)')
        return dummy_segmenter(r'C:\system32\rule34.pt')
    elif segmenter_name == 'DeepLabV3':
        if os.path.isfile(model_path) and model_path.split('.')[-1] in ['pt', 'pth']:
            return DeepLabV3_segmenter(model_path)
        else:
            raise RuntimeError(f'Path {model_path} doesnt exist :/. Did you download models? Did you even read README.md bi$ch?')
    else:
        raise RuntimeError(f'Model {segmenter_name} is not supported.')
