import cv2
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt,strip_optimizer,xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt
import json
import os 

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt",file="input",device='cpu',view_img=False,
        save_conf=False,line_thickness = 3,hide_labels=False, hide_conf=True):

    
    try:
        os.mkdir("output")
    except:
        pass
    
    device = select_device(opt.device) #select device
    half = device.type != 'cpu'

    model = attempt_load(poseweights, map_location=device)  #Load model
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    for source in os.listdir(file):
        frame_count = 0  #count no of frames
        source = 'input/'+source
        if source.isnumeric() :    
            cap = cv2.VideoCapture(int(source))    #pass video to videocapture object
        else :
            cap = cv2.VideoCapture(source)    #pass video to videocapture object
    
        if (cap.isOpened() == False):   #check if videocapture not opened
            print(source)
            print('Error while trying to read video. Please check path again')
            raise SystemExit()

        else:
            frame_width = int(cap.get(3))  #get video frame width
            frame_height = int(cap.get(4)) #get video frame height

            
            vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0] #init videowriter
            resize_height, resize_width = vid_write_image.shape[:2]
            out_video_name = f"{source.split('/')[-1].split('.')[0]}"
            out_video_name = f"output/{out_video_name}"

            while(cap.isOpened): #loop until cap opened or video not complete
            
                print("Frame {} Processing".format(frame_count))
                
                ret, frame = cap.read()  #get frame and success from video capture
                
                if ret: #if success is true, means frame exist
                    orig_image = frame #store frame
                    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) #convert frame to RGB
                    image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                    image = transforms.ToTensor()(image)
                    image = torch.tensor(np.array([image.numpy()]))
                
                    image = image.to(device)  #convert image data to device
                    image = image.float() #convert image to float precision (cpu)
                
                    with torch.no_grad():  #get predictions
                        output_data, _ = model(image)

                    output_data = non_max_suppression_kpt(output_data,   #Apply non max suppression
                                                0.25,   # Conf. Threshold.
                                                0.65, # IoU Threshold.
                                                nc=model.yaml['nc'], # Number of classes.
                                                nkpt=model.yaml['nkpt'], # Number of keypoints.
                                                kpt_label=True)
                
                    output = output_to_keypoint(output_data)
                    with open(out_video_name + '.json', 'a') as f:
                        json.dump({f"frame {frame_count}":output.tolist()}, f, indent=4)

                    frame_count += 1  #increment frame count
                else:
                    break

        cap.release()
        # cv2.destroyAllWindows()

        



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--file', type=str, default='input', help='video/0 for webcam') #video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    parser.add_argument('--view-img', action='store_true', help='display results')  #display results
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') #save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)') #box linethickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels') #box hidelabel
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences') #boxhideconf
    opt = parser.parse_args()
    return opt

#function for plot fps and time comparision graph
def plot_fps_time_comparision(time_list,fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparision Graph')
    plt.plot(time_list, fps_list,'b',label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparision_pose_estimate.png")
    

#main function
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device,opt.poseweights)
    main(opt)
