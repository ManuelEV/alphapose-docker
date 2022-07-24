"""Script for single-gpu/multi-gpu demo."""
import os
import platform
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
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.writer import DataWriter


class Args:
    # def __init__(self, checkpoint, sp, detector, detfile, indir, inputlist,
    #              inputimg, outputpath, save_img, vis, showbox, profile,
    #              format, min_box_area, detbatch, posebatch, eval, gpus,
    #              qsize, flip, debug, video, webcam, save_video, vis_fast,
    #              pose_flow, pose_track):
    def __init__(self):
        self.checkpoint = 'pretrained_models/fast_res50_256x192.pth'
        self.sp = False
        self.detector = 'yolo'
        self.detfile = ''
        self.inputpath = 'examples/demo/'
        # self.inputpath = ''
        self.inputlist = ''
        # self.inputimg = 'examples/demo/1.jpg'
        self.inputimg = ''
        self.outputpath = 'examples/res/'
        self.save_img = False
        self.vis = False
        self.showbox = False
        self.profile = False
        self.format = 'xd'
        self.min_box_area = 0
        self.detbatch = 5
        self.posebatch = 64
        self.eval = False
        self.gpus = '-1'
        self.qsize = 1024
        self.flip = False
        self.debug = False
        self.video = ''
        self.webcam = -1
        self.save_video = False
        self.vis_fast = False
        self.pose_flow = False
        self.pose_track = False
        self.cfg = update_config('configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml')

        self.gpus = [int(i) for i in self.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
        self.device = torch.device("cuda:" + str(self.gpus[0]) if self.gpus[0] >= 0 else "cpu")
        self.detbatch = self.detbatch * len(self.gpus)
        self.posebatch = self.posebatch * len(self.gpus)
        self.tracking = self.pose_track or self.pose_flow or self.detector == 'tracker'


class DemoInference:
    def __init__(self):
        print('WIIIIIII')

        self.args = Args()
        """----------------------------- Demo options -----------------------------"""

        if platform.system() == 'Windows':
            self.args.sp = True

        if not self.args.sp:
            torch.multiprocessing.set_start_method('forkserver', force=True)
            torch.multiprocessing.set_sharing_strategy('file_system')

        self.pose_model = None
        self.pose_dataset = None
        self.tracker = None

    def print_finish_info(self):
        print('===========================> Finish Model Running.')
        if (self.args.save_img or self.args.save_video) and not self.args.vis_fast:
            print('===========================> Rendering remaining images in the queue...')
            print(
                '===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')

    def process_image(self):
        self.args.inputpath = 'examples/input/'
        # for images
        if len(self.args.inputpath) or len(self.args.inputlist) or len(self.args.inputimg):
            inputpath = self.args.inputpath
            inputlist = self.args.inputlist
            inputimg = self.args.inputimg

            if len(inputlist):
                im_names = open(inputlist, 'r').readlines()
            elif len(inputpath) and inputpath != '/':
                for root, dirs, files in os.walk(inputpath):
                    im_names = files
                im_names = natsort.natsorted(im_names)
            elif len(inputimg):
                inputpath = os.path.split(inputimg)[0]
                im_names = [os.path.split(inputimg)[1]]
        print('IMG NAMES:')
        print(im_names)
        mode = 'image'
        input_source = im_names

        det_loader = DetectionLoader(input_source, get_detector(self.args), self.args.cfg, self.args,
                                     batchSize=self.args.detbatch,
                                     mode=mode,
                                     queueSize=self.args.qsize)
        det_worker = det_loader.start()
        runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }

        # Init data writer
        queueSize = 2 if mode == 'webcam' else self.args.qsize
        # if self.args.save_video and mode != 'image':
        #     from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt
        #     if mode == 'video':
        #         video_save_opt['savepath'] = os.path.join(self.args.outputpath,
        #                                                   'AlphaPose_' + os.path.basename(input_source))
        #     else:
        #         video_save_opt['savepath'] = os.path.join(self.args.outputpath,
        #                                                   'AlphaPose_webcam' + str(input_source) + '.mp4')
        #     video_save_opt.update(det_loader.videoinfo)
        #     writer = DataWriter(self.args.cfg, self.args, save_video=True, video_save_opt=video_save_opt,
        #                         queueSize=queueSize).start()
        # else:
        writer = DataWriter(self.args.cfg, self.args, save_video=False, queueSize=queueSize).start()

        # if mode == 'webcam':
        #     print('Starting webcam demo, press Ctrl + C to terminate...')
        #     sys.stdout.flush()
        #     im_names_desc = tqdm(self.args.loop())
        # else:
        data_len = det_loader.length
        im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

        batchSize = self.args.posebatch
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
                    if self.args.profile:
                        ckpt_time, det_time = getTime(start_time)
                        runtime_profile['dt'].append(det_time)
                    # Pose Estimation
                    inps = inps.to(self.args.device)
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
                        hm_j = self.pose_model(inps_j)
                        if flip:
                            hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], self.pose_dataset.joint_pairs,
                                                     shift=True)
                            hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                        hm.append(hm_j)
                    hm = torch.cat(hm)
                    if self.args.profile:
                        ckpt_time, pose_time = getTime(ckpt_time)
                        runtime_profile['pt'].append(pose_time)
                    if self.args.pose_track:
                        boxes, scores, ids, hm, cropped_boxes = track(self.tracker, self.args, orig_img, inps, boxes,
                                                                      hm,
                                                                      cropped_boxes, im_name, scores)
                    hm = hm.cpu()
                    writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                    if self.args.profile:
                        ckpt_time, post_time = getTime(ckpt_time)
                        runtime_profile['pn'].append(post_time)

                if self.args.profile:
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
            if self.args.sp:
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

    def load(self):
        self.pose_model = builder.build_sppe(self.args.cfg.MODEL, preset_cfg=self.args.cfg.DATA_PRESET)
        print('Loading pose model from %s...' % (self.args.checkpoint,))
        self.pose_model.load_state_dict(torch.load(self.args.checkpoint, map_location=self.args.device))
        self.pose_dataset = builder.retrieve_dataset(self.args.cfg.DATASET.TRAIN)
        if self.args.pose_track:
            self.tracker = Tracker(tcfg, self.args)
        if len(self.args.gpus) > 1:
            self.pose_model = torch.nn.DataParallel(self.pose_model, device_ids=self.args.gpus).to(self.args.device)
        else:
            self.pose_model.to(self.args.device)

        if len(self.args.gpus) > 1:
            self.pose_model = torch.nn.DataParallel(self.pose_model, device_ids=self.args.gpus).to(self.args.device)
        else:
            self.pose_model.to(self.args.device)
        self.pose_model.eval()
        print('Model loaded successfully')
