import multiprocessing as mp
from typing import Callable, Optional, Tuple

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import torch.autograd
import torch.nn as nn
import tqdm
from skimage.filters import gaussian
from skorch.net import NeuralNet
from torch.utils.data import Dataset


class FiveCropRestorer:
    WIDTH_BEFORE_CROP = 258
    HEIGHT_BEFORE_CROP = 144
    CROPPED_SIZE = 128

    @classmethod
    def restore_imgs(cls, batch_of_images: np.ndarray) -> np.ndarray:
        b, crops, c, h, w = batch_of_images.shape
        output_images = np.zeros(
            (
                len(batch_of_images),
                c,
                FiveCropRestorer.HEIGHT_BEFORE_CROP,
                FiveCropRestorer.WIDTH_BEFORE_CROP
            ),
            dtype=np.float32)

        for i, img in enumerate(batch_of_images):
            img = batch_of_images[i]
            output_images[i] = cls._restore_single(img)
        return output_images

    @classmethod
    def _restore_single(cls, img: np.ndarray) -> np.ndarray:
        crops, c, h, w = img.shape
        new = np.zeros((crops, c, FiveCropRestorer.HEIGHT_BEFORE_CROP,
                        FiveCropRestorer.WIDTH_BEFORE_CROP))
        mask = np.zeros((crops, 1, FiveCropRestorer.HEIGHT_BEFORE_CROP,
                         FiveCropRestorer.WIDTH_BEFORE_CROP))
        w = FiveCropRestorer.WIDTH_BEFORE_CROP
        crop_w = FiveCropRestorer.CROPPED_SIZE
        h = FiveCropRestorer.HEIGHT_BEFORE_CROP
        crop_h = FiveCropRestorer.CROPPED_SIZE

        new[0, :, :crop_h, :crop_w] = img[0]
        new[1, :, :crop_h, w - crop_w:w] = img[1]
        new[2, :, h - crop_h:h, :crop_w] = img[2]
        new[3, :, h - crop_h:h, w - crop_w:w] = img[3]

        mask[0, :, :crop_h, :crop_h] = 1
        mask[1, :, :crop_h, w - crop_w:w] = 1
        mask[2, :, h - crop_h:h, :crop_w] = 1
        mask[3, :, h - crop_h:h, w - crop_w:w] = 1

        i = int(round((h - crop_h) / 2.))
        j = int(round((w - crop_w) / 2.))
        new[4, :, i:i + crop_h, j:j + crop_w] = img[4]
        mask[4, :, i:i + crop_h, j:j + crop_w] = 1

        new = new.sum(axis=0)
        mask = mask.sum(axis=0)

        return new / (mask + 1e-8)


class GuidedBackprop:
    def __init__(self, net: NeuralNet, iterations: int):
        self.net = net
        self.iterations = iterations

        self._forward_relu_outputs = []
        self._handlers = []

    def gradient_output_zeroing_hook(self, module: nn.Module,
                                     grad_input: Tuple[torch.Tensor],
                                     grad_output: Tuple[torch.Tensor]):
        relu_output = self._forward_relu_outputs.pop(-1)
        relu_output[relu_output > 0] = 1
        grad_out = relu_output * torch.clamp(grad_input[0], min=0.0)
        return grad_out,

    def forward_output_saving_hook(self, module: nn.Module, x, y):
        self._forward_relu_outputs.append(y)

    def register_backward_hook_recursively(self):
        def apply_recursively(this: "GuidedBackprop", module_: nn.Module):
            for child in module_.children():
                apply_recursively(this, child)
            if isinstance(module_, nn.ReLU):
                handle = module_.register_backward_hook(this.gradient_output_zeroing_hook)
                this._handlers.append(handle)

        module = self.net.module_.extractor
        apply_recursively(self, module)

    def register_forward_hook_recursively(self):
        def apply_recursively(this: "GuidedBackprop", module_: nn.Module):
            for child in module_.children():
                apply_recursively(this, child)
            if isinstance(module_, nn.ReLU):
                handle = module_.register_forward_hook(this.forward_output_saving_hook)
                this._handlers.append(handle)

        module = self.net.module_.extractor
        apply_recursively(self, module)

    def unregister_hooks(self):
        for handle in self._handlers:
            handle.remove()

    def explain(self,
                data: Dataset,
                target_extractor: Optional[
                    Callable[[torch.Tensor], torch.Tensor]
                ] = None,
                explanation_postprocess: Optional[
                    Callable[[np.ndarray], np.ndarray]
                ] = None,
                show_img: bool = False) -> np.ndarray:
        self.register_forward_hook_recursively()
        self.register_backward_hook_recursively()

        self.net.module_ = self.net.module_.eval()
        dataset = self.net.get_dataset(data)
        iterator = self.net.get_iterator(dataset, training=True)
        pbar = tqdm.tqdm(total=len(iterator))
        for Xi, _ in iterator:
            Xi = Xi.to(self.net.device)
            Xi.requires_grad = True
            self.net.module_.zero_grad()
            yp = self.net.module_(Xi)
            if target_extractor is not None:
                yp = target_extractor(yp)

            predicted_class = torch.argmax(yp, dim=1).detach().cpu().numpy()[0]
            yp[:, predicted_class].backward()
            explanation = Xi.grad.detach()

            explanation = explanation.detach().cpu().numpy()
            Xi = Xi.detach().cpu().numpy()
            if explanation_postprocess is not None:
                explanation = explanation_postprocess(explanation)
                Xi = explanation_postprocess(Xi)

            confidence = torch.softmax(
                yp, dim=1
            ).detach().cpu().numpy()[0, predicted_class]
            combined = self.display_explanation(
                Xi, explanation, predicted_class, confidence, show_img
            )

            yield explanation, combined, predicted_class, confidence

            pbar.update(1)
        pbar.close()
        self.unregister_hooks()

    @classmethod
    def display_explanation(cls, original_img: np.ndarray, explanation: np.ndarray,
                            predicted_class: np.ndarray,
                            confidence: np.ndarray,
                            show_img: bool) -> np.ndarray:
        cm = plt.get_cmap("jet")
        explanation = convert_to_grayscale(explanation[0])
        explanation = cm(explanation[0])[..., :-1]

        original_img = original_img[0].transpose((1, 2, 0))

        new_img = 0.6 * original_img + .5 * explanation
        new_img = (new_img - new_img.min()) / (new_img.max() - new_img.min() + 1e-8) * 255
        new_img = new_img.astype(np.uint8)

        original_img = (original_img - original_img.min()) / (
                original_img.max() - original_img.min() + 1e-8) * 255
        original_img = imutils.resize(original_img.astype(np.uint8), height=680,
                                      inter=cv2.INTER_AREA)

        new_img = imutils.resize(new_img, height=680, inter=cv2.INTER_AREA)

        fig, ax = plt.subplots(2, 1, sharex=True)
        fig.tight_layout()

        ax[0].set_title("Class: {} | Confidence: {:.4f}".format(predicted_class,
                                                                confidence))
        ax[0].imshow(new_img)
        ax[1].imshow(original_img)
        ax[0].axis('off')
        ax[1].axis('off')

        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8) \
            .reshape((int(height), int(width), 3))
        if show_img:
            plt.show()
        return image


def testing(data_folder: str, model_path: str):
    from model import PretrainedModel
    from collections import defaultdict
    import imageio
    import common

    from dataset import UsgDataset
    from transformers import get_test_transformers
    from itertools import chain
    from pathlib import Path

    net = NeuralNet(
        PretrainedModel,
        criterion=nn.CrossEntropyLoss,
        module__extract_intermediate_values=False,
        module__n_dropout_runs=common.N_DROPOUT_INFERENCES,
        iterator_valid__shuffle=False,
        iterator_train__shuffle=False,
        iterator_valid__num_workers=mp.cpu_count(),
        iterator_valid__batch_size=1,
        iterator_train__batch_size=1,
        device="cuda",
    )
    net.initialize()
    net.load_params(f_params=model_path)

    data_paths = list(
        chain(
            (Path(data_folder) / "train" / "0").glob("*"),
            (Path(data_folder) / "train" / "1").glob("*")
        )
    )
    data_paths = list(sorted(data_paths, key=lambda x: int(x.name)))

    classes = [int(path.parent.name) for path in data_paths]
    _, valid_paths = common.get_train_test_split_from_paths(data_paths, classes)

    valid_dataset = UsgDataset(valid_paths, True,
                               transforms=get_test_transformers(),
                               has_crops=True)

    restorer = FiveCropRestorer()
    explainer = GuidedBackprop(net, 5)

    saves_path = Path("explanations")

    (saves_path / "0").mkdir(parents=True, exist_ok=True)
    (saves_path / "1").mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)

    for i, explanation_data in enumerate(explainer.explain(
            valid_dataset,
            target_extractor=lambda x: x[0].mean(dim=1),
            explanation_postprocess=restorer.restore_imgs,
            show_img=False
    )):
        name = valid_paths[i].parent.name

        explanation, combined, cls, confidence = explanation_data
        imageio.imwrite(
            saves_path / name / "{}_{}_{:.4f}.png".format(
                counts[name], cls, confidence
            ),
            combined
        )
        counts[name] += 1


def convert_to_grayscale(im_as_arr: np.ndarray) -> np.ndarray:
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99.5)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = gaussian(grayscale_im)
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("model_path")

    args = parser.parse_args()
    testing(args.data_path, args.model_path)
