import cv2

from paint_mask import MaskPainter
from move_mask import MaskMover
from poisson_image_editing import poisson_edit

#import argparse
import getopt
import sys
from os import path


def usage():
    print("Usage: python main.py [options] \n\n\
    Options: \n\
    \t-h\tPrint a brief help message and exits..\n\
    \t-s\t(Required) Specify a source image.\n\
    \t-t\t(Required) Specify a target image.\n\
    \t-m\t(Optional) Specify a mask image with the object in white and other part in black, ignore this option if you plan to draw it later.")


if __name__ == '__main__':
    # parse command line arguments
    args = {}
    
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "hs:t:m:p:")
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        print("See help: main.py -h")
        exit(2)
    for o, a in opts:
        if o in ("-h"):
            usage()
            exit()
        elif o in ("-s"):
            args["source"] = a
        elif o in ("-t"):
            args["target"] = a
        elif o in ("-m"):
            args["mask"] = a        
        else:
            assert False, "unhandled option"
    
    #     
    if ("source" not in args) or ("target" not in args):
        usage()
        exit()
    
    #    
    source = cv2.imread(args["source"])
    target = cv2.imread(args["target"])
    
    if source is None or target is None:
        print('Source or target image not exist.')
        exit()

    if source.shape[0] > target.shape[0] or source.shape[1] > target.shape[1]:
        print('Source image cannot be larger than target image.')
        exit()

    # draw the mask
    mask_path = ""
    if "mask" not in args:
        print('Please highlight the object to disapparate.\n')
        mp = MaskPainter(args["source"])
        mask_path = mp.paint_mask() 
    else:
        mask_path = args["mask"]
    
    # adjust mask position for target image
    print('Please move the object to desired location to apparate.\n')
    mm = MaskMover(args["target"], mask_path)
    offset_x, offset_y, target_mask_path = mm.move_mask()            

    # blend
    print('Blending ...')
    target_mask = cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE) 
    offset = offset_x, offset_y

    poisson_blend_result = poisson_edit(source, target, target_mask, offset)
    
    cv2.imwrite(path.join(path.dirname(args["source"]), 'target_result.png'), 
                poisson_blend_result)
    
    print('Done.\n')
