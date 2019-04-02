"""Utility script for ASAP."""
import argparse
import copy
from itertools import product
from joblib import Parallel, delayed
from lxml import etree
from pathlib import Path
import numpy as np
import cv2
import openslide
from openslide.deepzoom import DeepZoomGenerator


class Setup(object):

    def __init__(self):
        self.get_args()
        self.read_img()
        self.read_annotation()
        self.read_classes()
        self.check_outdir()

    def get_args(self):
        parser = argparse.ArgumentParser(
                            description="Utility script for ASAP.")
        parser.add_argument("wsi",
                            help="Whole Slide Image to use.")
        parser.add_argument("-s", "--size",
                            help="Width and height of patches.",
                            default=254)
        parser.add_argument("-ov", "--overlap",
                            help="Overlap size of each patches.",
                            default=1)
        parser.add_argument("-a", "--annotation",
                            help="Annotation file corresponding to the wsi.")
        parser.add_argument("-od", "--output_dir",
                            help="Where to save the output files.")
        parser.add_argument("-t", "--thresh",
                            help="Threshold value whether to cut out a patch.",
                            default="1.")
        parser.add_argument("-c", "--classes",
                            help="Add rule file to define on/off of classes.\
                                  if no rules selected, this targets all areas.")
        parser.add_argument("-sm", "--save_mask",
                            help="Whether to save masks.",
                            action="store_true")
        parser.add_argument("-m", "--mode",
                            choices=["rect", "non_rect", "all"],
                            default="all",
                            help="Shapes to extract.")
        self.args = parser.parse_args()

        self.wsi = Path(self.args.wsi)
        self.patch_size_no_overlap = int(self.args.size)  # 254
        self.overlap = int(self.args.overlap)  # 1
        self.patch_size_with_overlap = self.patch_size_no_overlap + self.overlap * 2  # 256
        self.thresh = float(self.args.thresh)  # 1.0
        self.save_mask = self.args.save_mask  # False
        self.modes = {"rect": ["Rectangle"],
                      "non_rect": ["Polygon", "Spline", "PointSet"],
                      "all": ["Rectangle", ]}

    def read_img(self):
        self.img = openslide.OpenSlide(str(self.wsi))
        self.dzimg = DeepZoomGenerator(self.img,
                                       self.patch_size_no_overlap,
                                       self.overlap)
        self.patch_area = self.patch_size_with_overlap ** 2
        self.deepest_level = self.dzimg.level_count - 1
        self.tiles = self.dzimg.level_tiles[-1]
        self.patch_iterator = product(list(range(self.tiles[0])), list(range(self.tiles[1])))
        self.width, self.height = self.img.dimensions

    def read_annotation(self):
        """
        One annotation should looks like
        {'Name': 'Annotation 0',
         'Type': 'Polygon',
         'PartOfGroup':'polygon',
         'Color': '#F4FA58'}

        """
        if self.args.annotation is None:
            annotation_path = str(self.wsi.parent/self.wsi.stem) + ".xml"
        else:
            annotation_path = self.args.annotation
        tree = etree.parse(annotation_path)
        self.annotations = tree.xpath("/ASAP_Annotations/Annotations/Annotation")
        self.annotation_groups = tree.xpath("/ASAP_Annotations/AnnotationGroups/Group")
        assert len(self.annotations) > 0, "Found no annotations."

    def read_classes(self):
        """
        self.classes is a list containing classes to cut out.
        ex: ["malignant", "stroma"]

        self.exclude_cls is a dict containint classes to exclude from its parent.
        ex: {"malignant": ["stroma", "blood_vessel"], "stroma": ["blood_vessel"]}
        """
        self.classes = []
        self.exclude_cls = {}
        if self.args.classes is not None:
            with open(self.args.classes, "r") as f:
                for i in f.readlines():
                    splitted = i.strip().split(" ")
                    cls, exclude = splitted[0], splitted[1:]
                    if i != "":
                        self.classes.append(cls)
                        self.exclude_cls[cls] = exclude
        else:
            # The leading line means "all classes".
            self.classes = [i.attrib['Name'] for i in self.annotation_groups]

    def check_outdir(self):
        if self.args.output_dir is None:
            path = self.wsi
            self.outdir = Path(path.parent/path.stem)
        else:
            self.outdir = Path(self.args.output_dir)
        self._check(self.outdir)
        self._check(self.outdir/"masks")
        self._check(self.outdir/"masks_thumb")
        self._check(self.outdir/"masks_saved")

        for group in self.annotation_groups:
            groupdir = self.outdir/group.attrib['Name']
            self._check(groupdir)
        print(f"Save files in {self.outdir}")

    def _check(self, path):
        if not path.exists():
            path.mkdir(parents=True)


class MaskMaker(Setup):

    def __init__(self):
        super().__init__()

    def make_mask(self):
        # Make base masks.
        self.masks = {i: np.zeros((self.height, self.width), dtype=np.uint8) for i in self.classes}
        for annotation in self.annotations:  # for applying rules.
            group = annotation.attrib["PartOfGroup"]  # for applying rules.
            if group in self.classes:
                contour = []
                for point in annotation.xpath("Coordinates/Coordinate"):
                    x = np.int32(np.float(point.attrib["X"]))
                    y = np.int32(np.float(point.attrib["Y"]))
                    contour.append([[x, y]])
                contour = np.array(contour, dtype=np.int32)
                self.masks[group] = cv2.drawContours(self.masks[group], [contour], 0, True, thickness=cv2.FILLED)

        # Excluding process
        if self.args.classes is not None:
            self.masks_exclude = copy.deepcopy(self.masks)
            for group in self.classes:
                for exclude in self.exclude_cls[group]:
                    cover = cv2.bitwise_and(self.masks[group], self.masks[exclude])
                    self.masks_exclude[group] = cv2.bitwise_xor(self.masks[group], cover)
            self.masks = self.masks_exclude

        self.save_mask_thumb()
        if self.args.save_mask:
            self.save_mask_large()

    def save_mask_thumb(self):
        if self.height > self.width:
            size = (int(512*self.width/self.height), 512)
        else:
            size = (512, int(512*self.height/self.width))
        for group, mask in self.masks.items():
            mask_thumb = cv2.resize(mask, size)
            cv2.imwrite(f"{self.outdir}/masks_thumb/mask_{group}.png", mask_thumb*255)
        print("Thumbnails saved.")

    def save_mask_large(self):
        for group in self.classes:
            cv2.imwrite(f"{self.outdir}/masks/{group}.png",
                        self.masks[group], (cv2.IMWRITE_PXM_BINARY, 1))
        print("Masks saved.")


class PatchMaker(MaskMaker):

    def __init__(self):
        super().__init__()

    def make_patch_parallel(self):
        print(f"Processing {self.tiles[0]*self.tiles[1]} patches x {len(self.classes)} classes.")
        for group in self.classes:
            if self.args.mode == "non_rect" or self.args.mode == "all":
                if np.sum(self.masks[group]) > 0:
                    print(f"{'-'*10} {group} {'-'*10}")
                    iterator = product(list(range(self.tiles[0])), list(range(self.tiles[1])))
                    parallel = Parallel(n_jobs=-1, verbose=1, backend="threading")
                    parallel([delayed(self.make_patch)(x, y, group) for x, y in iterator])
            elif self.args.mode == "rect":
                print("here")
                self.make_rect()

    def make_rect(self):
        for annotation in self.annotations:
            shapetype = annotation.attrib["Type"]
            group = annotation.attrib["PartOfGroup"]
            save_to = Path(f"{self.outdir}/{group}_rect")
            self._check(save_to)
            if group in self.classes and shapetype == "Rectangle":
                contour = []
                for point in annotation.xpath("Coordinates/Coordinate"):
                    x = np.int32(np.float(point.attrib["X"]))
                    y = np.int32(np.float(point.attrib["Y"]))
                    contour.append([x, y])
                contour = np.array(contour)
                minx = np.min(contour[:, 0])
                miny = np.min(contour[:, 1])
                maxx = np.max(contour[:, 0])
                maxy = np.max(contour[:, 1])
                rect = self.img.read_region((minx, miny), 0, (maxx - minx, maxy - miny))
                rect.save(str(save_to/f"{minx}_{miny}_{maxx}_{maxy}".png))

    def make_patch(self, x, y, group):
        if self.is_onshore(x, y, group):
            patch = self.dzimg.get_tile(self.deepest_level, (x, y))
            patch.save(f"{self.outdir}/{group}/{x:04}_{y:04}.png")

    def is_onshore(self, x, y, group):
        location, level, size = self.dzimg.get_tile_coordinates(self.deepest_level, (x, y))
        mask = self.masks[group][location[1]:location[1]+size[1], location[0]:location[0]+size[0]]
        patch_area = mask.shape[0] * mask.shape[1]
        if 1. < np.sum(mask) / patch_area:
            print(x, y, "Something went wrong.")
        if 1. >= np.sum(mask) / patch_area >= self.thresh:
            onshore = True
        else:
            onshore = False
        return onshore

    def saved_areas(self):
        for group in self.classes:
            baseimg = self.masks[group] * 255
            patches = []
            for path in list((self.outdir/group).glob("*.png")):
                coord = path.stem.split("_")
                patches.append(coord)
            for patch in patches:
                x, y = patch
                offsetx = self.patch_size_no_overlap * int(x)
                offsety = self.patch_size_no_overlap * int(y)
                target = baseimg[offsety:offsety+self.patch_size_with_overlap, offsetx:offsetx+self.patch_size_with_overlap]
                new_patch = np.ones(target.shape) * 127
                baseimg[offsety:offsety+self.patch_size_with_overlap, offsetx:offsetx+self.patch_size_with_overlap] = new_patch
            if self.height > self.width:
                size = (int(512*self.width/self.height), 512)
            else:
                size = (512, int(512*self.height/self.width))
            resized = cv2.resize(baseimg, size)
            # inverted = cv2.bitwise_not(resized)
            cv2.imwrite(f"{self.outdir}/masks_saved/mask_{group}.png", resized)
        print("Saved processed areas.")


if __name__ == '__main__':
    pm = PatchMaker()
    pm.make_mask()
    pm.make_patch_parallel()
    pm.saved_areas()
