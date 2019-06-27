import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, Dataset


class PartialConvolution(nn.Module):

    def __init__(self,
                 in_filters: int,
                 out_filters: int,
                 kernel_size: int,
                 stride: int = 1,
                 pad_size: int = 0):
        super().__init__()

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.kernel_size = kernel_size
        self.pad_size = pad_size
        self.stride = stride

        self.main_convo = nn.Conv2d(self.in_filters, self.out_filters, self.kernel_size,
                                    self.stride, self.pad_size, bias=True)
        self.mask_convo = nn.Conv2d(self.in_filters, self.out_filters, self.kernel_size,
                                    self.stride, self.pad_size, bias=False)
        self.mask_convo.weight.requires_grad = False
        self.mask_convo.weight.fill_(1)

    def forward(self, x) -> (torch.Tensor, torch.Tensor):
        img, mask = x
        partial = img * mask
        ones = torch.ones_like(mask)

        ones = self.mask_convo(ones)
        mask = self.mask_convo(mask)

        where_zeros = (mask < 1e-5).float()
        mask = where_zeros * 1e16 + (1 - where_zeros) * mask

        result = self.main_convo(partial) * ones / mask
        mask = torch.clamp(mask, 0, 1)
        return result, mask


class PartialWrapper(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()

        self.layer = layer

    def forward(self, x):
        return self.layer(x[0]), x[1]


class UpSamplingLayerPartialWrapper(nn.Module):
    def __init__(self, up_sampling_layer: nn.Upsample):
        super().__init__()

        self.layer = up_sampling_layer
        self.mask_layer = nn.UpsamplingNearest2d(scale_factor=self.layer.scale_factor)

    def forward(self, x):
        return self.layer(x[0]), self.mask_layer(x[1])


class Unet(nn.Module):
    def __init__(self, init_channels, out_channels):
        super().__init__()

        self.n_u = self.n_d = [64, 64, 64, 64, 64]
        self.k_u = self.k_d = [3, 3, 3, 3, 3]
        self.n_s = [4, 4, 4, 4, 4]
        self.k_s = [1, 1, 1, 1, 1]
        self.init_channels = init_channels
        self.out_channels = out_channels

        self.coding_layers = nn.ModuleList()
        self.decoding_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()

        next_filters = self.init_channels
        decoder_input_filters = []

        for i in range(len(self.n_u)):
            self.coding_layers.append(
                self._encoder_block(next_filters, self.n_u[i], self.k_u[i])
            )
            self.skip_layers.append(
                self._skip_block(self.n_u[i], self.n_s[i], self.k_s[i])
            )

            next_filters = self.n_u[i]
            decoder_input_filters.append(
                self.n_u[i] + self.n_s[i]
            )

        decoder_input_filters = decoder_input_filters[::-1]
        for i in range(len(self.n_d)):
            self.decoding_layers.append(
                self._decoder_block(decoder_input_filters[i], self.n_d[i], self.k_d[i])
            )

        self.out_layer = nn.Sequential(
            PartialConvolution(self.n_d[-1], self.out_channels, 1),
            PartialWrapper(nn.Sigmoid())
        )

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> (
            torch.Tensor, torch.Tensor
    ):
        skip_results = []
        for i, layer in enumerate(self.coding_layers):
            img, mask = layer((img, mask))
            skip_results.append(self.skip_layers[i]((img, mask)))

        skip_results = skip_results[::-1]

        for i, layer in enumerate(self.decoding_layers):
            img = torch.cat((img, skip_results[i][0]), dim=1)
            mask = torch.cat((mask, skip_results[i][1]), dim=1)
            img, mask = layer((img, mask))

        return self.out_layer((img, mask))

    @classmethod
    def _encoder_block(cls, in_channels, filters, kernel_size):
        layers = nn.Sequential(
            PartialConvolution(in_channels, filters, kernel_size, 2, kernel_size // 2),
            PartialWrapper(nn.InstanceNorm2d(filters)),
            PartialWrapper(nn.LeakyReLU(inplace=True)),
            PartialConvolution(filters, filters, kernel_size, 1, kernel_size // 2),
            PartialWrapper(nn.InstanceNorm2d(filters)),
            PartialWrapper(nn.LeakyReLU(inplace=True))
        )
        return layers

    @classmethod
    def _decoder_block(cls, in_channels, filters, kernel_size):
        layers = nn.Sequential(
            PartialWrapper(nn.InstanceNorm2d(in_channels)),
            PartialConvolution(in_channels, filters, kernel_size, 1, kernel_size // 2),
            PartialWrapper(nn.InstanceNorm2d(filters)),
            PartialWrapper(nn.LeakyReLU(inplace=True)),
            PartialConvolution(filters, filters, 1, 1, 0),
            PartialWrapper(nn.InstanceNorm2d(filters)),
            PartialWrapper(nn.LeakyReLU(inplace=True)),
            UpSamplingLayerPartialWrapper(nn.UpsamplingNearest2d(scale_factor=2))
        )
        return layers

    @classmethod
    def _skip_block(cls, in_channels, filters, kernel_size):
        layers = nn.Sequential(
            PartialConvolution(in_channels, filters, kernel_size, 1, 0),
            PartialWrapper(nn.InstanceNorm2d(filters)),
            PartialWrapper(nn.LeakyReLU(inplace=True))
        )
        return layers


def fit_img(img: np.ndarray) -> np.ndarray:
    def get_min_max_cord(img: np.ndarray, axis: int) -> Tuple[int, int]:
        img = img.astype(np.float32)
        line = img.sum(axis=axis)
        line[line > 0] = 1
        grad = np.diff(line)
        a_min = np.argmax(grad).item()
        a_max = np.argmin(grad).item()
        return a_min, a_max

    y_min, y_max = get_min_max_cord(img, 1)
    x_min, x_max = get_min_max_cord(img, 0)
    return img[y_min:y_max, x_min:x_max]


def get_biggest_component(img: np.ndarray) -> np.ndarray:
    binary = (img > 0).astype(np.uint8) * 255
    connectivity = 8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity,
                                                                    cv2.CV_16U)
    areas = stats[:, cv2.CC_STAT_AREA]
    indices = labels == np.argsort(-areas)[1]
    img[~indices] = 0
    return img


def get_clean_and_mask_images(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img = get_biggest_component(img)
    img = fit_img(img)
    new_h, new_w = match_dims_to_be_divisible(img)
    img = cv2.resize(img, (new_w, new_h))

    fl_img = img.astype(np.float32)

    hor_grad = fl_img[1:] - fl_img[:-1]
    ver_grad = fl_img[:, 1:] - fl_img[:, :-1]

    hor_grad = hor_grad[:, 1:]
    ver_grad = ver_grad[1:]

    grad = hor_grad + ver_grad
    grad = (grad - grad.min()) / (grad.max() - grad.min())
    grad = grad - grad.mean()
    grad = np.clip(grad, 0, 1)

    percentile = np.percentile(grad, q=95)
    masked = (grad > percentile).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    masked = cv2.morphologyEx(masked, cv2.MORPH_DILATE, kernel, iterations=1)
    masked = np.pad(masked, [[1, 0], [1, 0]], constant_values=0, mode="constant")

    connectivity = 4
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(masked, connectivity,
                                                                    cv2.CV_16U)
    areas = stats[:, cv2.CC_STAT_AREA]
    for i, area in enumerate(areas):
        if area < 256:
            indices = labels == i
            masked[indices] = 0

    return img, masked


def reconstruction_loss(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor
) -> torch.Tensor:
    diff = (y_pred - y_true) * mask
    loss = diff.abs().mean()
    return loss


def match_dims_to_be_divisible(an_img: np.ndarray) -> Tuple[int, int]:
    h, w = an_img.shape[0], an_img.shape[1]

    new_h = (h // 32 + 1) * 32
    new_w = (w // 32 + 1) * 32
    return new_h, new_w


class RestorationDataset(Dataset):
    def __init__(self, files: List[Path], verbose: bool):
        self.files = files
        self.verbose = verbose

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        path = self.files[index].as_posix()
        try:
            an_img = cv2.imread(path, 0)
            an_img, mask = get_clean_and_mask_images(an_img)
            mask = 1 - mask

            an_img = an_img.astype(np.float32) / 255
            mask = mask.astype(np.float32)
            an_img = an_img * mask
            an_img = an_img[np.newaxis]
            mask = mask[np.newaxis]
        except:
            an_img = np.zeros((1, 1, 1), dtype=np.float32)
            mask = np.ones_like(an_img)

        return an_img, mask

    def __len__(self) -> int:
        return len(self.files)

    def _filter_out_paths(self):
        new_files: List[Path] = []
        before = len(self.files)
        for path in tqdm.tqdm(self.files, disable=not self.verbose):
            try:
                an_img = cv2.imread(path.as_posix(), 0)
                _ = get_clean_and_mask_images(an_img)
                new_files.append(path)
            except Exception:
                pass
        after = len(new_files)
        if self.verbose:
            print("Removed: {} files".format(before - after))
        self.files = new_files


def infinite_dataloader(loader: DataLoader):
    while True:
        for data in loader:
            yield data


def train(
        data_path: str,
        save_image_path: str,
        output_model_path: str
):
    data_path = Path(data_path)
    assert data_path.exists()

    save_image_path = Path(save_image_path)
    assert save_image_path.exists()

    output_model_path = Path(output_model_path)
    output_model_path.mkdir()

    batch_size = 1

    rng = np.random.RandomState(0)

    lower_images = list(data_path.rglob("lower.png"))
    testing_samples = rng.choice(lower_images, size=20)

    dataset = RestorationDataset(lower_images, True)
    testing_dataset = RestorationDataset(testing_samples, False)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=mp.cpu_count(), pin_memory=True)
    test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=mp.cpu_count(), pin_memory=True)

    unet = Unet(1, 1)
    optimiser = optim.Adam(unet.parameters(), lr=0.001)
    pbar = tqdm.tqdm(total=20 * len(loader))

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        unet = unet.cuda()
    loader = infinite_dataloader(loader)

    for i in range(pbar.total):
        images, masks = next(loader)
        if images.shape[-1] == 1:
            pbar.update(1)
            continue
        if use_cuda:
            images = images.cuda()
            masks = masks.cuda()
        optimiser.zero_grad()
        prediction, _ = unet(images, masks)
        loss = reconstruction_loss(prediction, images, masks)
        loss.backward()
        optimiser.step()
        pbar.update(1)
        pbar.set_postfix({"loss": "%.7f" % loss.cpu().item()})

        if i > 0 and i % 30 == 0:
            validate(test_loader, unet, use_cuda, save_image_path)
            torch.save(unet.state_dict(), (output_model_path / "weights.pt").as_posix())


@torch.no_grad()
def validate(
        loader: DataLoader,
        model: nn.Module,
        use_cuda: bool,
        save_images_path: Path
) -> float:
    val_loss = 0
    index = 0
    samples = 0
    for images, masks in loader:
        if images.shape[-1] == 1:
            continue
        if use_cuda:
            images = images.cuda()
            masks = masks.cuda()
        prediction, _ = model(images, masks)
        loss = reconstruction_loss(prediction, images, masks)
        val_loss += loss.cpu().item()
        samples += len(prediction)

        prediction = np.clip(
            (prediction.detach().cpu().numpy() * 255).astype(np.uint8),
            0, 255
        )
        images = np.clip(
            (images.detach().cpu().numpy() * 255).astype(np.uint8),
            0, 255
        )

        to_show = np.concatenate([prediction, images], axis=-1)
        for to_save in to_show:
            try:
                cv2.imwrite(
                    (save_images_path / f"{index}.png").as_posix(),
                    to_save[0]
                )
                index += 1
            except cv2.error:
                continue

    print("Val loss: {:.7f}".format(val_loss / samples))
    return val_loss


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Script for training image recovery using learnable deep image priors"
    )
    parser.add_argument("data_path")
    parser.add_argument("save_image_path")
    parser.add_argument("output_model_path")

    args = parser.parse_args()

    train(
        args.data_path,
        args.save_image_path,
        args.output_model_path
    )
