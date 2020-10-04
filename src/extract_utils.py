__all__ = ['Settings', 'PersonImagesExtractor']


from DetectorAPI import DetectorAPI
#odapi = DetectorAPI(path_to_ckpt=DETECTION_MODEL_PATH)

# Cell
class Settings:

    def __init__(self, score_threshold = 0.5, frame_capture_rate_sec=60., min_height_to_width_ratio = 0.0,
                 save_intermediate=False, keep_intermediate_in_memory=False):
        """`score_threshold` -- порог отсечения для детектора.
        Захват картинки происходит каждые `frame_capture_rate_sec` секунд.
        `min_height_to_width_ratio` -- минимальное соотношение высоты и ширины картинки человека. Позволяет отфильтровать сидячих людей.
        Если `save_intermediate` True, то все изображения сохраняются в промежуточную коллекцию.
        """
        self.score_threshold = score_threshold
        self.frame_capture_rate_sec = frame_capture_rate_sec
        self.min_height_to_width_ratio = min_height_to_width_ratio
        self.save_intermediate = save_intermediate
        self.keep_intermediate_in_memory = keep_intermediate_in_memory

import os
import pickle
import sys
import time
import uuid
import warnings
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import cv2
from tqdm import tqdm

class PersonImagesExtractor:
    def __init__(self, detector : DetectorAPI, settings : Settings = None):
        if not isinstance(detector, DetectorAPI):
            raise ValueError(f"Argument 'detector' must be DetectorAPI instance.")
        self.detector = detector
        if settings == None:
            settings = Settings()
        self.settings = settings
        self.intermediate = {}

    def extract_crop_images_and_boxes(self, img) -> Tuple[List,List]:
        crop_images = []
        crop_boxes = []
        boxes, scores, classes, _ = self.detector.processFrame(img)
        for box, score, class_ in zip(boxes, scores, classes):
            # Class 1 represents human
            if class_ == 1 and score >= self.settings.score_threshold:
                (xA, yA, xB, yB) = box
                if (xB-xA)/(yB-yA) >=  self.settings.min_height_to_width_ratio:
                    crop_image = img[xA:xB,yA:yB,:]
                    crop_images.append(crop_image.astype(np.uint8))
                    crop_boxes.append(np.array(box, dtype=np.int16))
        return crop_images, crop_boxes

    def extract_all_images(self, video_filename : Union[str,Path], out_images_folder : Union[str,Path], 
        save_images_on_the_fly : bool = True, intermediate_folder:str=None, start_since_sec:int = 0, finish_at_sec:Union[int,None]=None) -> None:

        if not os.path.exists(video_filename):  
            print(f"File {video_filename} could not be found. Extraction will be missed.")  
            return None
        if not os.path.exists(out_images_folder):
            os.mkdir(out_images_folder)
            warnings.warn(f"Folder {out_images_folder} didn't exist. It had to create it.")

        
        def build_image_name():
            return  f"{os.path.splitext(os.path.basename(video_filename))[0]}_{frame:06d}f_{i:02d}p_{uuid.uuid4().hex[:4]}.jpg"
        #for video_filename in videos:
        if True: #TODO
            start=time.time()
            cap = cv2.VideoCapture(video_filename)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT )
            frames_per_sec = cap.get(cv2.CAP_PROP_FPS)
            durationSec = frame_count / frames_per_sec

            intermediate = {}
            
            ini_frame = int(start_since_sec*frames_per_sec)
            if finish_at_sec == None:
                fin_frame = frame_count
            else:
                fin_frame = finish_at_sec*frames_per_sec
            delta_frame = int(self.settings.frame_capture_rate_sec*frames_per_sec)

            frame = ini_frame

            progress_bar = tqdm(total=fin_frame-ini_frame, file=sys.stdout)
            while frame < fin_frame:
                try:
                    #if int((sec/durationSec)*1000) % 100 == 0:
                    #    print(sec,'sec. Elapsed:', time.time()-start)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    success, img = cap.read()
                    if success:
                        #TODO: to think about using callback
                        crop_images, boxes = self.extract_crop_images_and_boxes(img)
                        for i, crop_image in enumerate(crop_images):

                            if save_images_on_the_fly:
                                filename = build_image_name()
                                filepath = os.path.join(out_images_folder, filename )
                                cv2.imwrite( filepath, crop_image)

                        if self.settings.keep_intermediate_in_memory:
                            intermediate[frame] = (crop_images, boxes)

                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        break

                except Exception as e:
                    print(video_filename, frame, e)
                frame += delta_frame

                progress_bar.update(delta_frame)
            self.intermediate = intermediate
            cap.release()
            tqdm.close(progress_bar)
            finish=time.time()
            print(f"\n{video_filename} has been processed for {(finish-start)/60.:.2f} minutes")
            if self.settings.save_intermediate:
                self.save_intermediate(intermediate_folder)

      
    def save_intermediate(self, intermediate_folder:Union[str, Path], file_prefix:str = '') -> Union[str,bytes]:
        if self.intermediate and intermediate_folder:
            if not os.path.exists(intermediate_folder):
                os.mkdir(intermediate_folder)
                warnings.warn(f"Intermidiate folder {intermediate_folder} didn't exist. It had to create it.")
            filename = f"{os.path.splitext(os.path.basename(file_prefix))[0]}_intermediate_{uuid.uuid4().hex[:4]}.pkl"
            filepath = os.path.join(intermediate_folder, filename )
            with open(filepath, 'wb') as file:
                pickle.dump(self.intermediate, file)
                print(f"Intermediate info is saved to {filepath}")
        return filepath
                
    def load_intermediate(self, filepath:Union[str,Path,bytes]) -> dict:
        with open(filepath, 'rb') as file:
            intermediate = pickle.load(file)
        return intermediate

            
print(f'Module {__file__} imported')
