import cv2
import numpy as np
import math
import os
import re


# read and scale down image
def crop_grains(image_name):
    m = re.search('[\w]+', image_name)
    folder_name = m.group(0)
    # folder_name = image_name[:-4]
    img = cv2.imread(image_name)
    gray = cv2.imread(image_name, 0)

    #	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # thresholded = cv2.cvtColor(th,cv2.COLOR_GRAY2BGR)

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(th, kernel, iterations=1)

    image, contours, hier = cv2.findContours(eroded, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_NONE)

    # th1 = cv2.cvtColor(th,cv2.COLOR_GRAY2BGR)
    i = 1

    #	print('Number of separate grains: ', len(contours))
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # path = os.getcwd()
    os.chdir(os.getcwd() + '/' + folder_name)
    for c in contours:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx_polygon = cv2.approxPolyDP(c, epsilon, True)

        if cv2.contourArea(c) >= 250:
            # cv2.drawContours(img, [approx_polygon], -1, (0, 255, 0), 3)

            # print('Contour length: ', cv2.arcLength(c, True))

            # Initialize mask
            # mask = np.zeros((gray.shape[0], gray.shape[1]))
            mask = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

            cv2.drawContours(mask, c, -1, (255, 255, 255), 1)

            # Create mask that defines the polygon of points
            cv2.fillConvexPoly(mask, c, 255, 1)
            mask = cv2.dilate(mask, kernel, iterations=2)

            # mask = mask.astype(np.bool)

            # Create output image (untranslated)
            # out = np.zeros_like(img)
            # out[mask] = gray[mask]

            out = cv2.bitwise_and(img, img, mask=mask)

            # Find centroid of polygon
            (meanx, meany) = approx_polygon[0].mean(axis=0)

            # Find centre of image
            (cenx, ceny) = (img.shape[1] / 2, img.shape[0] / 2)

            # Make integer coordinates for each of the above
            (meanx, meany, cenx, ceny) = np.floor([meanx, meany, cenx, ceny]).astype(np.int32)

            # Calculate final offset to translate source pixels to centre of image
            (offsetx, offsety) = (-meanx + cenx, -meany + ceny)

            # Define remapping coordinates
            (mx, my) = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
            ox = (mx - offsetx).astype(np.float32)
            oy = (my - offsety).astype(np.float32)

            # Translate the image to centre
            out_translate = cv2.remap(out, ox, oy, cv2.INTER_LINEAR)
            # cv2.namedWindow('out',cv2.WINDOW_NORMAL)
            # cv2.imshow('out',out_translate)
            # cv2.waitKey(900)

            # out_translate = cv2.resize(out_translate, (560,560))

            # out_translate_new = cv2.resize(out_translate, (int(224/0.38),int(224/0.38)))

            height = int(224 / 0.3)
            width = int(224 / 0.3)

            # h2 = int(0.7*224/0.38)

            out = out_translate[ceny + 30 - 200:ceny + 30 + 200, cenx - 200:cenx + 200]
            out1 = cv2.resize(out, (224, 224))

            outfile = "image%d.png" % (i + 1)
            cv2.imwrite(outfile, out1)
            #			print('image' + str(i+1) + ' copied to disk')

            i += 1

    # out1 = cv2.resize(img, (1000,500))
    # cv2.imshow('Otsu_out', out1)
    # cv2.waitKey(0)
    # os.chdir(path)
    cv2.destroyAllWindows()
