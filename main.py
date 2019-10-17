import itertools
import math
import os
import random
import sys

import cv2
import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import common

class PlateGenerator:
    def __init__(self):
        self.FONT_DIR = "./fonts"
        self.BACKGROUNDS_PATH =  "backgrounds/bw.jpg"
        self.FONT_HEIGHT = 64 # Pixel size to which the chars are resized
        self.OUTPUT_SHAPE = (240, 420)
        self.CHARS = common.CHARS + " "

        self.position_char_list = []
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def load_fonts(self, folder_path):
        """
        Load font 

        :return:

        fonts = List of fonts 
        font_char_ims = dict with key as 
        """
        font_char_ims = {}
        fonts = [f for f in os.listdir(folder_path) if f.endswith('.ttf')]

        for font in fonts:
            # Font path
            font_path = os.path.join(folder_path, font)
            
            font_char_ims[font] = dict(self.make_char_ims(font_path,
                                                    self.FONT_HEIGHT))
        return fonts, font_char_ims

    @classmethod 
    def euler_to_mat(cls, yaw, pitch, roll):
        # Rotate clockwise about the Y-axis
        c, s = math.cos(yaw), math.sin(yaw)

        M = np.matrix([[  c, 0.,  s],
                        [ 0., 1., 0.],
                        [ -s, 0.,  c]])

        # Rotate clockwise about the X-axis
        c, s = math.cos(pitch), math.sin(pitch)
        M = np.matrix([[ 1., 0., 0.],
                        [ 0.,  c, -s],
                        [ 0.,  s,  c]]) * M

        # Rotate clockwise about the Z-axis
        c, s = math.cos(roll), math.sin(roll)
        M = np.matrix([[  c, -s, 0.],
                        [  s,  c, 0.],
                        [ 0., 0., 1.]]) * M

        return M

    @classmethod
    def rounded_rect(cls, shape, radius):
        out = np.ones(shape)
        out[:radius, :radius] = 0.0
        out[-radius:, :radius] = 0.0
        out[:radius, -radius:] = 0.0
        out[-radius:, -radius:] = 0.0

        cv2.circle(out, (radius, radius), radius, 1.0, -1)
        cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)
        cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)
        cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)
        return out

    @classmethod
    def generate_code(cls):
        return "{}{}{}{}{}{}{}".format(
            random.choice(common.DIGITS),
            random.choice(common.DIGITS),
            random.choice(common.DIGITS),
            random.choice(common.DIGITS),
            random.choice(common.LETTERS),
            random.choice(common.LETTERS),
            random.choice(common.LETTERS))

    @classmethod
    def pick_colors(cls):
        first = True
        text_color = 0
        plate_color = 1
        while first or plate_color - text_color < 0.3:
            text_color = 0
            plate_color = 1
            if text_color > plate_color:
                text_color, plate_color = plate_color, text_color
            first = False

        return text_color, plate_color
    
    def get_random_number(self, custom_color=40):

        x = random.randrange(40, 100)/100
        return x
            
    def make_char_ims(self, font_path, output_height):
        """
        Create characters  images from font 

        :yields: Tuple
        c       = String with the character
        img_p   = Numpy array that represents the character as Image
        """
        font_size = output_height * 4

        font = ImageFont.truetype(font_path, font_size)

        # Get mas font size
        height = max(font.getsize(c)[1] for c in self.CHARS)

        # Iterate over the characters and generate image
        for c in self.CHARS:

            width = font.getsize(c)[0]
            im = Image.new("RGBA", (width, height), (0, 0, 0))

            r_color = int(255 * self.get_random_number())
            g_color = int(255 * self.get_random_number())

            draw = ImageDraw.Draw(im)
            draw.text((0, 0), c, (r_color, g_color, 255), font=font)
            scale = float(output_height) / height
            im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
            
            img_p = np.array(im)[:, :, 0].astype(np.float32) / 255.

            yield c, img_p



    def make_affine_transform(self, from_shape, to_shape, 
                            min_scale, max_scale,
                            scale_variation=1.0,
                            rotation_variation=1.0,
                            translation_variation=1.0):
        out_of_bounds = False

        from_size = np.array([[from_shape[1], from_shape[0]]]).T
        to_size = np.array([[to_shape[1], to_shape[0]+30]]).T

        scale = random.uniform((min_scale + max_scale) * 0.5 -
                            (max_scale - min_scale) * 0.5 * scale_variation,
                            (min_scale + max_scale) * 0.5 +
                            (max_scale - min_scale) * 0.5 * scale_variation)

        if scale > max_scale or scale < min_scale:
            out_of_bounds = True

        roll = random.uniform(-0.3, 0.3) * rotation_variation
        pitch = random.uniform(-0.2, 0.2) * rotation_variation
        yaw = random.uniform(-1.2, 1.2) * rotation_variation

        # Compute a bounding box on the skewed input image (`from_shape`).
        M = self.euler_to_mat(yaw, pitch, roll)[:2, :2]
        h, w = from_shape
        corners = np.matrix([[-w, +w, -w, +w],
                                [-h, -h, +h, +h]]) * 0.5
        skewed_size = np.array(np.max(M * corners, axis=1) -
                                np.min(M * corners, axis=1))

        # Set the scale as large as possible such that the skewed and scaled shape
        # is less than or equal to the desired ratio in either dimension.
        scale *= np.min(to_size / skewed_size)

        # Set the translation such that the skewed and scaled image falls within
        # the output shape's bounds.
        trans = (np.random.random((2,1)) - 0.5) * translation_variation

        trans = ((2.0 * trans) ** 5.0) / 2.0
        if np.any(trans < -0.5) or np.any(trans > 0.5):
            out_of_bounds = True

        trans = (to_size - skewed_size * scale) * trans

        center_to = to_size / 2.
        center_from = from_size / 2.

        M = self.euler_to_mat(yaw, pitch, roll)[:2, :2]
        M *= scale
        M = np.hstack([M, trans + center_to - M * center_from])

        return M, out_of_bounds

    def generate_plate(self, font_height, char_ims):
        """
        Generate Plate from char_ims
        """
        h_padding = random.uniform(0.1, 0.15) * font_height
        v_padding = random.uniform(0.0, 0.0) * font_height

        spacing = font_height * random.uniform(-0.05, 0.05)

        radius = 1 + int(font_height * 0.1 * random.random())

        code = self.generate_code()

        text_width = sum(char_ims[c].shape[1] for c in code)
        text_width += (len(code) - 1) * spacing

        out_shape = (int(font_height + v_padding * 1),
                    int(text_width + h_padding * 2))
    
        text_color, plate_color = self.pick_colors()
        text_color = 0

        text_mask = np.zeros(out_shape)
        print(f"text mask shape {text_mask.shape}")

        x = h_padding
        y = v_padding
        
        for c in code:

            # Get image
            char_im = char_ims[c]
            
            print(f"CHARIMAGE SHAPE {char_im.shape}")
            
            ix, iy = int(x), int(y)
            deltaY = iy + char_im.shape[0]
            deltaX = ix + char_im.shape[1]
            # print(".............")
            # print(f"IY is {iy}")
            # print(f"IX is {ix}")
            # print(f"DELTA Y {deltaY}")
            # print(f"DELTA X {deltaX}")
            
            p1 = [ix, iy]
            p2 = [deltaX, deltaY]

            # print(f"P1 {p1}, P2 {p2}")

            char_pos  = {c : [p1, p2] }

            self.position_char_list.append(char_pos)

            text_mask[p1[1]: p2[1], p1[0]:p2[0]] = char_im

            update_x_position = char_im.shape[1] + spacing

            x += update_x_position

        plate = (np.ones(out_shape) * plate_color * (1. - text_mask) +
                np.ones(out_shape) * text_color * text_mask)
        
        # Return numpy array and String with plate code
        rounded_rect, code_string = self.rounded_rect(out_shape, radius), code.replace(" ", "")
        # cv2.imwrite("super/super.jpg", plate*255.)
        
        return plate, rounded_rect, code_string,

    def generate_bg(self):

        fname = self.BACKGROUNDS_PATH

        bg = (cv2.imread(fname, 0)) / 255.

        # if (bg.shape[1] >= self.OUTPUT_SHAPE[1] and
        #     bg.shape[0] >= self.OUTPUT_SHAPE[0]):
        #     print("COMPATIBLE BACKGROUND")
            
        # x = random.randint(0, bg.shape[1] - self.OUTPUT_SHAPE[1])
        # y = random.randint(0, bg.shape[0] - self.OUTPUT_SHAPE[0])

        # print(x, y)
        
        # bg = bg[y:y + self.OUTPUT_SHAPE[0], x:x + self.OUTPUT_SHAPE[1]]

        return bg

    def generate_im(self, char_ims):
        """
        Generate images based on char_to_images

        char_ims: Dict with Keys as String "character", Values as numpy arrays (64,31)

        """
        #print(f"CHHAR IMAGES keys: {char_ims.keys}")
        # for key, value in char_ims.items():
        #     print (key)
        #     print(value.shape)
        #     cv2.imwrite("super/test.jpg", value*255)
        #     break

        bg = self.generate_bg()

        plate_or, plate_mask, code = self.generate_plate(self.FONT_HEIGHT, char_ims)
        
        M, out_of_bounds = self.make_affine_transform(
                                from_shape=plate_or.shape,
                                to_shape=bg.shape,
                                min_scale=0.9,
                                max_scale=1.0,
                                rotation_variation=0.1,
                                scale_variation=0.0,
                                translation_variation=0.0)
        print(f"M SHAPE {M.shape}")
        print(M)
        plate = cv2.warpAffine(plate_or, M, (bg.shape[1], bg.shape[0]))
        cv2.imwrite("super/plate.jpg", plate*255.)

        print(f"WAR APIINE PLATE plate shape {plate.shape}")

        plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))
        print(f"WAR APIINE PLATE plate_mask shape {plate.shape}")
        cv2.imwrite("super/plate_mask.jpg", plate_mask*255.)

        out = plate * plate_mask + bg * (1.0 - plate_mask)

        ## out = cv2.resize(out, (self.OUTPUT_SHAPE[1]+50, self.OUTPUT_SHAPE[0]))

        out += np.random.normal(scale=0.05, size=out.shape)
        out = np.clip(out, 0.0, 1.)
        
        w = out.shape[1]
        h = out.shape[0]

        print(f"W {w} AND H {h}")
        plate_y_shape, plate_x_shape = plate_or.shape
        print(f"PLATE X {plate_x_shape} Y SHAPE {plate_y_shape}" )

        for d in  self.position_char_list:
            for k,v in d.items():
                p1, p2 = v

                print(f"P1 {p1} and P2 {p2}")
                new_pos_1 =(p1[0] + (w - plate_x_shape ), p1[1] + (plate_y_shape)) 
                new_pos_2 =(p2[0] + (w - plate_x_shape ), p2[1] + (plate_y_shape))

                print(f"P1_new {new_pos_1} and P2_new {new_pos_2}")
                # print(new_pos_1, new_pos_2)
                # # cv2.putText(out, k , tuple(p2), self.font, 4,(0,255,255),2, cv2.LINE_AA)
                cv2.rectangle(out, new_pos_1, new_pos_2, (0,255,255), 2)
                
        return out, code, not out_of_bounds

    def generate_ims(self):
        """
        Generate number plate images.

        :return:
            Iterable of number plate images.
        """
        fonts, font_char_ims = self.load_fonts(self.FONT_DIR)
        random_font = random.choice(fonts)

        while True:
            yield self.generate_im(font_char_ims[random_font])


if __name__ == "__main__":
    OUTPUT_FOLDER = "out"
    try:
        os.mkdir(OUTPUT_FOLDER)
    except:
        print('directory already exist...')
    
    plate_generator = PlateGenerator()

    im_gen = itertools.islice(plate_generator.generate_ims(), int(sys.argv[1]))

    for img_idx, (im, plate, p) in enumerate(im_gen):
        fname = "{}/00000{:0d}_{}_{}.png".format(OUTPUT_FOLDER,
                                                img_idx,
                                                plate,
                                                "1" if p else "0")
        print (fname)
        cv2.imwrite(fname, im * 255.)