class ImageStandardizer:
    from PIL import Image
    import numpy as np    
    def __init__(self, max_thumb_size):
        self.max_thumb_size = max_thumb_size
    def array_from_file(self, filepath):
        from PIL import Image
        import numpy as np   
        with Image.open(filepath) as image:
            blank_image = Image.new('RGB', self.max_thumb_size, 'black')
            image.thumbnail(self.max_thumb_size, Image.BICUBIC)
            blank_image.paste(image, (0,0)) # add to left upper corner
            return np.asarray(blank_image)
    def array_from_file_list(self, filepaths):
        from PIL import Image
        import numpy as np  
        res=[]
        for filepath in filepaths:
            with Image.open(filepath) as image:
                blank_image = Image.new('RGB', self.max_thumb_size, 'black')
                image.thumbnail(self.max_thumb_size, Image.BICUBIC)
                blank_image.paste(image, (0,0)) # add to left upper corner
                res.append(np.asarray(blank_image))
        return np.asarray(res)
    def resize_array(self, np_image):
        import cv2
        import numpy as np
        res = cv2.resize(np_image, dsize=self.max_thumb_size, interpolation=cv2.INTER_CUBIC)    
        return res
    def save_to_image_file(self, image, output_image_path):
        image.save(output_image_path)      