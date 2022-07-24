"""Script for single-gpu/multi-gpu demo."""
import argparse
import os
import platform
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
import natsort

from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter


class DemoInference:
    def __init__(self):
        """----------------------------- Demo options -----------------------------"""
        # parser = argparse.ArgumentParser(description='AlphaPose Demo')
        # parser.add_argument('--cfg', type=str, required=True,
        #                     help='experiment configure file name')

        # parser.add_argument('--checkpoint', type=str, required=True,
        #                     help='checkpoint file name')
        self.checkpoint = 'pretrained_models/fast_res50_256x192.pth'
        # parser.add_argument('--sp', default=False, action='store_true',
        #                     help='Use single process for pytorch')
        self.sp = False
        # parser.add_argument('--detector', dest='detector',
        #                     help='detector name', default="yolo")
        self.detector = 'yolo'
        # parser.add_argument('--detfile', dest='detfile',
        #                     help='detection result file', default="")
        self.detfile = []
        # parser.add_argument('--indir', dest='inputpath',
        #                     help='image-directory', default="")
        self.indir = 'examples/demo/'
        # parser.add_argument('--list', dest='inputlist',
        #                     help='image-list', default="")
        self.inputlist = ''
        # parser.add_argument('--image', dest='inputimg',
        #                     help='image-name', default="")
        self.inputimg = ''
        # parser.add_argument('--outdir', dest='outputpath',
        #                     help='output-directory', default="examples/res/")
        self.outputpath = 'examples/res/'
        # parser.add_argument('--save_img', default=False, action='store_true',
        #                     help='save result as image')
        self.save_img = False
        # parser.add_argument('--vis', default=False, action='store_true',
        #                     help='visualize image')
        self.vis = False
        # parser.add_argument('--showbox', default=False, action='store_true',
        #                     help='visualize human bbox')
        self.showbox = False
        # parser.add_argument('--profile', default=False, action='store_true',
        #                     help='add speed profiling at screen output')
        self.profile = False
        # parser.add_argument('--format', type=str,
        #                     help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
        self.format = 'xd'
        # parser.add_argument('--min_box_area', type=int, default=0,
        #                     help='min box area to filter out')
        self.min_box_area = 0
        # parser.add_argument('--detbatch', type=int, default=5,
        #                     help='detection batch size PER GPU')
        self.detbatch = 5
        # parser.add_argument('--posebatch', type=int, default=64,
        #                     help='pose estimation maximum batch size PER GPU')
        self.posebatch = 64
        # parser.add_argument('--eval', dest='eval', default=False, action='store_true',
        #                     help='save the result json as coco format, using image index(int) instead of image name(str)')
        self.eval = False
        # parser.add_argument('--gpus', type=str, dest='gpus', default="0",
        #                     help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
        self.gpus = '0'
        # parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
        #                     help='the length of result buffer, where reducing it will lower requirement of cpu memory')
        self.qsize = 1024
        # parser.add_argument('--flip', default=False, action='store_true',
        #                     help='enable flip testing')
        # NOTA: dar otro nombre a flip
        self.flip = False
        # parser.add_argument('--debug', default=False, action='store_true',
        #                     help='print detail information')
        self.debug = False

        # """----------------------------- Video options -----------------------------"""
        # parser.add_argument('--video', dest='video',
        #                     help='video-name', default="")
        self.video = ''
        # parser.add_argument('--webcam', dest='webcam', type=int,
        #                     help='webcam number', default=-1)
        self.webcam = -1
        # parser.add_argument('--save_video', dest='save_video',
        #                     help='whether to save rendered video', default=False, action='store_true')
        self.save_video = False
        # parser.add_argument('--vis_fast', dest='vis_fast',
        #                     help='use fast rendering', action='store_true', default=False)
        self.vis_fast = False
        # """----------------------------- Tracking options -----------------------------"""
        # parser.add_argument('--pose_flow', dest='pose_flow',
        #                     help='track humans in video with PoseFlow', action='store_true', default=False)
        self.pose_flow = False
        # parser.add_argument('--pose_track', dest='pose_track',
        #                     help='track humans in video with reid', action='store_true', default=False)
        self.pose_track = False

        self.cfg = update_config('configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml')

        if platform.system() == 'Windows':
            self.sp = True

        self.gpus = [int(i) for i in self.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
        self.device = torch.device("cuda:" + str(self.gpus[0]) if self.gpus[0] >= 0 else "cpu")
        self.detbatch = self.detbatch * len(self.gpus)
        self.posebatch = self.posebatch * len(self.gpus)
        self.tracking = self.pose_track or self.pose_flow or self.detector == 'tracker'

        if not self.sp:
            torch.multiprocessing.set_start_method('forkserver', force=True)
            torch.multiprocessing.set_sharing_strategy('file_system')

    # args = parser.parse_args()

    # cfg = update_config(cfg)

    def check_input(self):
        # for wecam
        if self.webcam != -1:
            detbatch = 1
            return 'webcam', int(self.webcam)

        # for video
        if len(self.video):
            if os.path.isfile(self.video):
                videofile = self.video
                return 'video', videofile
            else:
                raise IOError('Error: --video must refer to a video file, not directory.')

        # for detection results
        if len(self.detfile):
            if os.path.isfile(self.detfile):
                detfile = self.detfile
                return 'detfile', detfile
            else:
                raise IOError('Error: --detfile must refer to a detection json file, not directory.')

        # for images
        if len(self.inputpath) or len(self.inputlist) or len(self.inputimg):
            inputpath = self.inputpath
            inputlist = self.inputlist
            inputimg = self.inputimg

            if len(inputlist):
                im_names = open(inputlist, 'r').readlines()
            elif len(inputpath) and inputpath != '/':
                for root, dirs, files in os.walk(inputpath):
                    im_names = files
                im_names = natsort.natsorted(im_names)
            elif len(inputimg):
                inputpath = os.path.split(inputimg)[0]
                im_names = [os.path.split(inputimg)[1]]

            return 'image', im_names

        else:
            raise NotImplementedError

    def print_finish_info(self):
        print('===========================> Finish Model Running.')
        if (self.save_img or self.save_video) and not self.vis_fast:
            print('===========================> Rendering remaining images in the queue...')
            print(
                '===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')

    def loop(self):
        n = 0
        while True:
            yield n
            n += 1

    def load_model(self):
        mode, input_source = self.check_input()

        if not os.path.exists(self.outputpath):
            os.makedirs(self.outputpath)

        # Load detection loader
        if mode == 'webcam':
            det_loader = WebCamDetectionLoader(input_source, get_detector(args), self.cfg, args)
            det_worker = det_loader.start()
        elif mode == 'detfile':
            det_loader = FileDetectionLoader(input_source, self.cfg, args)
            det_worker = det_loader.start()
        else:
            det_loader = DetectionLoader(input_source, get_detector(args), self.cfg, args, batchSize=self.detbatch,
                                         mode=mode,
                                         queueSize=self.qsize)
            det_worker = det_loader.start()

        # Load pose model
        pose_model = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)

        print('Loading pose model from %s...' % (self.checkpoint,))
        pose_model.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
        pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
        if self.pose_track:
            tracker = Tracker(tcfg, args)
        if len(self.gpus) > 1:
            pose_model = torch.nn.DataParallel(pose_model, device_ids=self.gpus).to(self.device)
        else:
            pose_model.to(self.device)
        pose_model.eval()

        runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }

        # Init data writer
        queueSize = 2 if mode == 'webcam' else self.qsize
        if self.save_video and mode != 'image':
            from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt
            if mode == 'video':
                video_save_opt['savepath'] = os.path.join(self.outputpath,
                                                          'AlphaPose_' + os.path.basename(input_source))
            else:
                video_save_opt['savepath'] = os.path.join(self.outputpath,
                                                          'AlphaPose_webcam' + str(input_source) + '.mp4')
            video_save_opt.update(det_loader.videoinfo)
            writer = DataWriter(self.cfg, args, save_video=True, video_save_opt=video_save_opt,
                                queueSize=queueSize).start()
        else:
            writer = DataWriter(self.cfg, args, save_video=False, queueSize=queueSize).start()

        if mode == 'webcam':
            print('Starting webcam demo, press Ctrl + C to terminate...')
            sys.stdout.flush()
            im_names_desc = tqdm(self.loop())
        else:
            data_len = det_loader.length
            im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

        batchSize = self.posebatch
        if flip:
            batchSize = int(batchSize / 2)
        try:
            for i in im_names_desc:
                start_time = getTime()
                with torch.no_grad():
                    (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                    if orig_img is None:
                        break
                    if boxes is None or boxes.nelement() == 0:
                        writer.save(None, None, None, None, None, orig_img, im_name)
                        continue
                    if self.profile:
                        ckpt_time, det_time = getTime(start_time)
                        runtime_profile['dt'].append(det_time)
                    # Pose Estimation
                    inps = inps.to(self.device)
                    datalen = inps.size(0)
                    leftover = 0
                    if (datalen) % batchSize:
                        leftover = 1
                    num_batches = datalen // batchSize + leftover
                    hm = []
                    for j in range(num_batches):
                        inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                        if flip:
                            inps_j = torch.cat((inps_j, flip(inps_j)))
                        hm_j = pose_model(inps_j)
                        if flip:
                            hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], pose_dataset.joint_pairs, shift=True)
                            hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                        hm.append(hm_j)
                    hm = torch.cat(hm)
                    if self.profile:
                        ckpt_time, pose_time = getTime(ckpt_time)
                        runtime_profile['pt'].append(pose_time)
                    if self.pose_track:
                        boxes, scores, ids, hm, cropped_boxes = track(tracker, args, orig_img, inps, boxes, hm,
                                                                      cropped_boxes, im_name, scores)
                    hm = hm.cpu()
                    writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                    if self.profile:
                        ckpt_time, post_time = getTime(ckpt_time)
                        runtime_profile['pn'].append(post_time)

                if self.profile:
                    # TQDM
                    im_names_desc.set_description(
                        'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                            dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']),
                            pn=np.mean(runtime_profile['pn']))
                    )
            self.print_finish_info()
            while (writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining ' + str(
                    writer.count()) + ' images in the queue...')
            writer.stop()
            det_loader.stop()
        except Exception as e:
            print(repr(e))
            print('An error as above occurs when processing the images, please check it')
            pass
        except KeyboardInterrupt:
            self.print_finish_info()
            # Thread won't be killed when press Ctrl+C
            if self.sp:
                det_loader.terminate()
                while (writer.running()):
                    time.sleep(1)
                    print('===========================> Rendering remaining ' + str(
                        writer.count()) + ' images in the queue...')
                writer.stop()
            else:
                # subprocesses are killed, manually clear queues

                det_loader.terminate()
                writer.terminate()
                writer.clear_queues()
                det_loader.clear_queues()
