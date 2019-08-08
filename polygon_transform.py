import cv2
import numpy as np
from skimage.measure import find_contours, approximate_polygon, \
    subdivide_polygon

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def isPolygonInPolygon(poly1,poly2):
    poly2 = Polygon(poly2)
    pointsInPoly = 0
    for poi in poly1:
        poi = Point(poi)
        # all points in polygon 
        if(poly2.contains(poi)):
            pointsInPoly += 1
    pointsInPolyFactor = pointsInPoly / len(poly1)
    if (pointsInPolyFactor >= 0.5 ):
        return True
    else: 
        return False
        # one point in polygon
        # if(poly2.contains(poi)):
        #     contains_point.append(True)

def polygonTransformHierarchy(polygon_list_raw,scr):
    polygon_list_raw = np.array(polygon_list_raw)
    dimension = polygon_list_raw.ndim
    polygon_list_hierarchy = []
    # if polygon list has only one element the dimension is 3
    # PIXELMAP TO POLYGON
    if((dimension == 1) or (dimension == 3)):
        polygon_list = polygon_list_raw
    # POLYGON TO PIXELMAP
    elif(dimension == 2):
        polygon_list = np.array([x[0] for x in polygon_list_raw])
    else:
        raise ValueError('Polygon list unexpected dimension')
    for i,polygon1 in enumerate(polygon_list):
        level = 0
        for polygon2 in polygon_list:
            if(isPolygonInPolygon(polygon1, polygon2)):
                level += 1
        if(level > len(polygon_list_hierarchy)-1):
            dif = (level+1)- len(polygon_list_hierarchy)
            for _ in range(dif):
                polygon_list_hierarchy.append([])   
                
        if (dimension == 2):
            polygon1 = np.array(polygon_list_raw[i])
        polygon_list_hierarchy[level].append(polygon1)
    return polygon_list_hierarchy


def pixelMapToPolygons(pixel_map, scr): 

    black_count = np.count_nonzero(pixel_map == 0)
    white_count = np.count_nonzero(pixel_map == 255)
    label_seagrass = None
    imgray = cv2.cvtColor(pixel_map, cv2.COLOR_BGR2GRAY)
    imgray = cv2.bitwise_not(imgray)

    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 6, True) for cnt in contours]
    polygon_list = []
    for i,  contour in enumerate(contours):
        
        area = cv2.contourArea(contour)
        if(area <1):
            continue
        point_list = []
        for points in contour:
            x = points[0][0] / 512
            y = points[0][1] / 256
            point_list.append([x,y])
        polygon = np.array(point_list)
        polygon_list.append(polygon)

    polygon_list_hierarchy = polygonTransformHierarchy(polygon_list, scr)
    new_polygon_list = []
    anno_types = []
    anno_labels = []
    for i,level in enumerate(polygon_list_hierarchy):

        if(i%2 == 0):
            test = 36
        else:
            test = 37
        for polygon in level:
            new_polygon_list.append(polygon)
            anno_labels.append(test)
            anno_types.append('polygon')
    return new_polygon_list, anno_types, anno_labels

def polygonsToPixelMap(polygon_list, height, width, scr):
    # init with background
    pixel_map = np.full((height, width), 255.0)
    polygon_list_hierarchy = polygonTransformHierarchy(polygon_list, scr)
    mult = np.array([width, height])
    for i, level in enumerate(polygon_list_hierarchy):
        for polygon in level:
            polygon_shape = np.array(polygon[0]) * mult
            polygon_shape= [np.array(polygon_shape, dtype=np.int32 )]
            if(polygon[2] == "Seagrass"):
                color = 0.0
            else:
                color = 255.0
            cv2.fillPoly(pixel_map, polygon_shape, color)

    return pixel_map
