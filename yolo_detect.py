#!/usr/bin/env python

"""
 YOLO v7
"""

import argparse
import time
from pathlib import Path

import cv2
import torch
from numpy import random
from threading import Lock

from models.experimental import attempt_load
from utils.datasets import LoadWebcam, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# Variables
pstanding_count = 0
pseated_count = 0
pfallen_count = 0
frame_count = 0

mutex = Lock()

def count(founded_classes,im0):
  model_values=[]
  aligns=im0.shape
  align_bottom=aligns[0]
  align_right=(aligns[1]/1.7 ) 

  for i, (k, v) in enumerate(founded_classes.items()):
    a=f"{k} = {v}"
    model_values.append(v)
    align_bottom=align_bottom-35
    cv2.putText(im0, str(a) ,(int(align_right),align_bottom), cv2.FONT_HERSHEY_SIMPLEX, 1,(45,255,255),1,cv2.LINE_AA)

def detect(save_img=False, opt=None):
    global pseated_count, pfallen_count, pstanding_count, mutex
    print(opt)
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    register = False
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size


    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()


    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        #view_img = check_imshow()
        #cudnn.benchmark = True  # set True to speed up constant image size inference
        #dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        dataset = LoadWebcam(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    archivo=open(str(save_dir)+"log.txt",'w')
    #l = [dataset.__next__]
    global frame_count
    for path, img, im0s, vid_cap in dataset:
        frame_count = frame_count + 1
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        if save_img:
            cv2.imwrite("./images/" + str(time.time()) + ".jpg", im0s)
            print(f" The image with the result is saved in ./images")
        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if False:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                x = ''
            else:
                p, s, x, im0, frame = path, '',' ', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            txt_path = str(save_dir / 'labels' / p.stem) + (f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                cont = [0,0,0]
                founded_classes ={}
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class                
                    class_index=int(c)
                    count_of_object=int(n)
                    
                    cont[int(c)] = int(n)
                    founded_classes[names[class_index]]=int(n)
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    x += f"{n},{names[int(c)]},"
                    if register: #ASFfo
                        count(founded_classes=founded_classes,im0=im0)  # Applying counter function

                # Write results
                mutex.acquire()
                pseated_count = pseated_count + cont[0]
                pfallen_count = pfallen_count + cont[1]
                pstanding_count = pstanding_count + cont[2]
                mutex.release()
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            if register: #ASF
                #-----------------------------------------
                # Get date of the image
                #-----------------------------------------
                img = pyexiv2.Image(path)
                date = img.read_exif()['Exif.Photo.DateTimeOriginal']

                # #----------------------------------------
                # # Save in a file: Date P_sen P_tumb P_Depie
                archivo.write(date+f';{cont[0]};{cont[1]};{cont[2]}'+'\n')
                print(cont)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    #parser.add_argument('--source', type=str, default='OT1225900_cropped_1.jpg', help='source')  # file/folder, 0 for webcam ASF:Dataset/images/test/  -fgl
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam ASF:Dataset/images/test/  -fgl
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.40, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.50, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect/datos_predictor', help='save results to project/name')  # fgl
    parser.add_argument('--name', default='prueba', help='save results to project/name')                          # fgl
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    #parser.add_argument('--registro', action='store_true', help='take log of detection and date')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()