### PROGRAM TO RESIZE THE SUPER FILE IMAGES INTO  178x218 
### 


# Importing libs
import cv2
from PIL import Image
import glob


#out path
outpath = 'super_resized/'
try:
	os.mkdir('super_resized')
except:
	print('patch exist')
counter = 0
for filename in glob.glob('super/*.png'):
	img = cv2.imread(filename)
	img = cv2.resize(img,(157,163))
	cv2.imwrite(outpath+'{}.bmp'.format(counter), img)
	counter += 1
print('done!!')
