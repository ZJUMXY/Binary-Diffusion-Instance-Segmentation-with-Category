import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as tnf
from torchvision.datasets import VisionDataset
from PIL import Image
import os.path


class COCODataset(VisionDataset):
    """ Transformation from original CocoDetection. Use annotation id to index. Pad all img&masks to 640*640"""

    def __init__(
            self,
            root: str,
            annFile: str,
    ) -> None:
        super().__init__(root)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.getAnnIds(iscrowd=False)))
        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")


    def __getitem__(self, index: int):
        id = self.ids[index]
        ann = self.coco.loadAnns(id)[0]
        image_id = ann['image_id']
        image = self._load_image(image_id)
        trans = transforms.ToTensor()
        image = trans(image)
        shape = image.shape
        image = tnf.pad(image, (0, 640 - shape[-1], 0, 640 - shape[-2]), mode="constant", value=0)
        masks = torch.tensor(self.coco.annToMask(ann))
        masks = tnf.pad(masks, (0, 640 - shape[-1], 0, 640 - shape[-2]), mode="constant", value=0)
        classes = self.json_category_id_to_contiguous_id[ann["category_id"]]
        return image, masks, classes

    def __len__(self) -> int:
        return len(self.ids)


if __name__ == "__main__":
    dataset = COCODataset(root="./dataset/val2017", annFile="./dataset/instances_val2017.json")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
    for i in range(8):
        x = dataset[i]
