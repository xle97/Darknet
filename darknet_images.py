import argparse
from configparser import Interpolation
import os
import glob
import random
import time
import cv2
import numpy as np
import darknet


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="/home/rzhang/Desktop/35",
                        help="image source. It can be a single image, a"
                        "txt with paths to them, or a folder. Image valid"
                        " formats are jpg, jpeg or png."
                        "If no input is given, ")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default="/home/rzhang/Desktop/second_AHGE/yolo_1375_final.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',default=True,
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--save_labels", action='store_true',default=True,
                        help="save detections bbox for each image in yolo format")
    parser.add_argument("--config_file", default="/home/rzhang/Desktop/second_AHGE/yolo_1375.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="/home/rzhang/Desktop/second_AHGE/collect.data",     #./cfg/coco.data
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.1,
                        help="remove detections with lower confidence")
    return parser.parse_args()


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))


def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))


def prepare_batch(images, network, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_images = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
    darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    return darknet.IMAGE(width, height, channels, darknet_images)

def letterbox_image(image, dst_size, pad_color = (114,114,114)):
    src_h, src_w = image.shape[:2]
    dst_h, dst_w = dst_size
    scale = min(dst_h / src_h, dst_w / src_w)
    pad_h, pad_w = int(round(src_h * scale)), int(round(src_w * scale))

    if image.shape[0:2] != (pad_w, pad_h):
        image_dst = cv2.resize(image, (pad_w, pad_h), interpolation=cv2.INTER_LINEAR)
    else:
        image_dst = image

    top = int((dst_h - pad_h) / 2)
    down = int((dst_h - pad_h + 1) / 2)
    left = int((dst_w - pad_w) / 2)
    right = int((dst_w - pad_w + 1) / 2)

    # add border
    image_dst = cv2.copyMakeBorder(image_dst, top, down, left, right, cv2.BORDER_CONSTANT, value=pad_color)

    # x_offset, y_offset = max(left, right) / dst_w, max(top, down) / dst_h
    return image_dst      #, x_offset, y_offset


def image_detection(image_or_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)   #必须要resize以匹配模型否则会预测乱
    
    # if type(image_or_path) == "str":
    #     image = cv2.imread(image_or_path)
    # else:
    #     image = image_or_path
    # image_or_path = '/home/rzhang/Desktop/hs/3_R10C18_3089_184_1.png'
    image = cv2.imread(image_or_path)

    h,w,_ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)


    # image_resized =  cv2.resize(image, (width, height),interpolation=cv2.INTER_LINEAR)
    # sized = cv2.cvtColor(image_resized,cv2.COLOR_BGR2GRAY)
    # img_in = np.expand_dims(sized, axis=0)
    # image_resized = img_in.transpose(1,2,0)
   

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors,h,w)
    # image = cv2.resize(image, (w, h),
    #                            interpolation=cv2.INTER_LINEAR)
    # return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections,[w,h]
    return image, detections, [w,h]


def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    image_height, image_width, _ = check_batch_shape(images, batch_size)
    darknet_images = prepare_batch(images, network)
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_obj(detections, num, len(class_names), nms)
        predictions = darknet.remove_negatives(detections, class_names, num)
        images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
        batch_predictions.append(predictions)
    darknet.free_batch_detections(batch_detections, batch_size)
    return images, batch_predictions


def image_classification(image, network, class_names):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
    # image_resized = image_rgb
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    darknet.free_image(darknet_image)
    return sorted(predictions, key=lambda x: -x[1])


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names, size):
    """
    Files saved with image_name.txt and relative coordinates
    """
    output_path = '/home/rzhang/Desktop/output/check_result.txt'
    # file_name = os.path.splitext(name)[0] + ".txt"
    file_name = os.path.splitext(name)[0].split('/')[-1] 
    r =  min(image.shape[1] /size[0], image.shape[0] / size[1])
    # scalew = size[0] / image.shape[1] 
    # scaleh = size[1] / image.shape[0]
    unpad_w = int(round(size[0] * r))
    unpad_h = int(round(size[1] * r))
    dw = image.shape[1] - unpad_w
    dh = image.shape[0] - unpad_h
    dw /=2
    dh /=2
    scale_w = size[0] / unpad_w
    scale_h = size[1] / unpad_h
    with open(output_path, "a") as f:
        f.write("{}=>[".format(file_name))
        for label, confidence, bbox in detections:
            # x, y, w, h = convert2relative(image, bbox)   ##不需要转相对值
            x,y,w,h=bbox
            x = x - w / 2
            y = y - h / 2
            ####letter_box
            # xx = x + w
            # yy = y + h
            # x  = (x - dw)  * scale_w
            # xx = (xx - dw) * scale_w
            # y  = (y - dh)  * scale_h
            # yy = (yy - dh) * scale_h
            # label=label.split('_')[0]
            # w = xx - x
            # h = yy - y

            
            # x2 = x + w/2
            # y2 = y + w/2
            # label = class_names.index(label)  不能直接转成label
            f.write("[{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}],".format(label,float(confidence)*0.01,x, y, w, h))
        f.write(']\n')


def batch_detection_example():
    args = parser()
    check_arguments_errors(args)
    batch_size = 3
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=batch_size
    )
    image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
    images = [cv2.imread(image) for image in image_names]
    images, detections,  = batch_detection(network, images, class_names,
                                           class_colors, batch_size=batch_size)
    for name, image in zip(image_names, images):
        cv2.imwrite(name.replace("data/", ""), image)
    print(detections)


def main():
    args = parser()
    check_arguments_errors(args)

    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )

    images = load_images(args.input)
    
    index = 0
    while True:
        # loop asking for new image paths if no list is given
        if args.input:
            if index >= len(images):
                break  
            # images = glob.glob("/home/rzhang/Desktop/hs/*.png")
            image_name = images[index]
            
        else:
            image_name = input("Enter Image Path: ")

        prev_time = time.time()
        image, detections ,size = image_detection(
            image_name, network, class_names, class_colors, args.thresh
            )
        if args.save_labels:
            save_annotations(image_name, image, detections, class_names, size)
        darknet.print_detections(detections, args.ext_output)
        fps = int(1/(time.time() - prev_time))
        print("FPS: {}".format(fps))
        new_name=image_name.split('/')[-1]
        img_path=os.path.join("/home/rzhang/Desktop/output2",new_name)
        cv2.imwrite(img_path,image)
        if not args.dont_show:
            cv2.imshow('Inference', image)
            if cv2.waitKey() & 0xFF == ord('q'):
                break
        
        index += 1


if __name__ == "__main__":
    # unconmment next line for an example of batch processing
    # batch_detection_example()
    main()
