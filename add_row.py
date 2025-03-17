# add a row to every png with odd height, because odd height pngs don't work
# with our video encoding
import cv2
import glob

pngs = glob.glob("output/*.png")
for png in pngs:
    im = cv2.imread(png)
    if im.shape[0] % 2 != 0:
        im = cv2.copyMakeBorder(im, 0, 1, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.imwrite(png, im)

