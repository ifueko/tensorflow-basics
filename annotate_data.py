import face_recognition
from matplotlib import pyplot as plt
import cv2
import os
from tqdm import tqdm
global count_michelle, count_kelly, count_beyonce, count_unknown

base_dir = 'videos'
michelle = 'michelle'
kelly = 'kelly'
beyonce = 'beyonce'
unknown = 'unknown'
train = 'train'
test = 'test'

os.makedirs(os.path.join(base_dir, train, michelle), exist_ok=True)
os.makedirs(os.path.join(base_dir, train, kelly), exist_ok=True)
os.makedirs(os.path.join(base_dir, train, beyonce), exist_ok=True)
os.makedirs(os.path.join(base_dir, train, unknown), exist_ok=True)
os.makedirs(os.path.join(base_dir, test, michelle), exist_ok=True)
os.makedirs(os.path.join(base_dir, test, kelly), exist_ok=True)
os.makedirs(os.path.join(base_dir, test, beyonce), exist_ok=True)
os.makedirs(os.path.join(base_dir, test, unknown), exist_ok=True)

train_data = [
    "raw/8_Days_of_Christmas.mp4",
    "raw/Bills_Bills_Bills.mp4",
    "raw/Bootylicious.mp4",
    "raw/Bug_A_Boo.mp4",
    "raw/Cater_2_U.mp4",
    "raw/Emotion.mp4",
    "raw/Get_On_The_Bus.mp4",
    "raw/Independent_Women_Part_1.mp4",
    "raw/Jumpin_Jumpin.mp4",
    "raw/Lose_My_Breath.mp4",
    "raw/Me_Myself_and_I.mp4",
    "raw/Nasty_Girl.mp4",
    "raw/No_No_No.mp4",
    "raw/Say_My_Name.mp4",
    "raw/Soldier.mp4",
]

test_data = [
    "raw/Stand_Up_For_Love.mp4",
    "raw/Survivor.mp4",
    "raw/With_Me_Part_1.mp4",
]
fig, ax = plt.subplots(2, 1)
ax[0].axis('off')
ax[1].axis('off')
plt.ion()
count_michelle = max([int(f.split('.')[-2].split('_')[-1]) for f in os.listdir(os.path.join(base_dir, train, michelle)) if f.split('.')[-1] == 'png'] + [0]) + 1
count_kelly = max([int(f.split('.')[-2].split('_')[-1]) for f in os.listdir(os.path.join(base_dir, train, kelly)) if f.split('.')[-1] == 'png'] + [0]) + 1
count_beyonce = max([int(f.split('.')[-2].split('_')[-1]) for f in os.listdir(os.path.join(base_dir, train, beyonce)) if f.split('.')[-1] == 'png'] + [0]) + 1
count_unknown = max([int(f.split('.')[-2].split('_')[-1]) for f in os.listdir(os.path.join(base_dir, train, unknown)) if f.split('.')[-1] == 'png'] + [0]) + 1
print(count_michelle, count_beyonce, count_kelly, count_unknown)
def annotate_video(path, train_=True, start=24):
  global count_michelle, count_kelly, count_beyonce, count_unknown
  cap = cv2.VideoCapture(path)
  success, image = cap.read()
  total = 0
  while success:
    total += 1
    success, image = cap.read()
  cap = cv2.VideoCapture(path)
  success, image = cap.read()
  pbar = tqdm(total=total)
  last_val = 'u'
  last = 'UNKNOWN'
  count = 0
  skip_count = start 
  while success:
    count += 1
    if count % skip_count != 0:
      success, image = cap.read()
      pbar.update(1)
      continue
    skip_count = 24
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    faces = face_recognition.face_locations(image)
    if len(faces) > 0:
      im_orig = image.copy()
      for y1, x1, y2, x2 in faces:
        face_im = im_orig[y1:y2, x2:x1].copy()
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        ax[0].imshow(face_im)
        ax[1].imshow(image)
        ax[0].set_title('Frame {} of {} (Guess: {})'.format(count, total, last))
        plt.show()
        new_val = input('Enter [m] for Michelle, [k] for Kelly, [b] for Beyonce, [u] for Other, [s] to skip (LAST: {} ({}): '.format(last, count))
        if len(new_val) == 0:
          val = last_val
        else:
          val = new_val[0]
        last_val = val
        if val == 'm':
          last = 'Michelle'
          save_file = os.path.join(base_dir, train if train_ else test, michelle, 'michelle_{}.png'.format(count_michelle))
          count_michelle += 1
        elif val == 'k':
          last = 'Kelly'
          save_file = os.path.join(base_dir, train if train_ else test, kelly, 'kelly_{}.png'.format(count_kelly))
          count_kelly += 1
        elif val == 'b':
          last = 'Beyonce'
          save_file = os.path.join(base_dir, train if train_ else test, beyonce, 'beyonce_{}.png'.format(count_beyonce))
          count_beyonce += 1
        elif val == 'u':
          last = 'UNKNOWN'
          save_file = os.path.join(base_dir, train if train_ else test, unknown, 'unknown_{}.png'.format(count_unknown))
          count_unknown += 1
        else:
          print('SKIPPED')
          skip_count = 64 
          continue
        print("    SAVING {}".format(save_file))
        plt.imsave(save_file, face_im)
    success, image = cap.read()
    pbar.update(1)

for video in train_data:
  start  = 24 
  if type(video) != str:
    video, start = video
  path = os.path.join(base_dir, video)
  annotate_video(path, start=start)

count_michelle = max([int(f.split('.')[-2].split('_')[-1]) for f in os.listdir(os.path.join(base_dir, test, michelle)) if f.split('.')[-1] == 'png'] + [0]) + 1
count_kelly = max([int(f.split('.')[-2].split('_')[-1]) for f in os.listdir(os.path.join(base_dir, test, kelly)) if f.split('.')[-1] == 'png'] + [0]) + 1
count_beyonce = max([int(f.split('.')[-2].split('_')[-1]) for f in os.listdir(os.path.join(base_dir, test, beyonce)) if f.split('.')[-1] == 'png'] + [0]) + 1
count_unknown = max([int(f.split('.')[-2].split('_')[-1]) for f in os.listdir(os.path.join(base_dir, test, unknown)) if f.split('.')[-1] == 'png'] + [0]) + 1
print(count_michelle, count_beyonce, count_kelly, count_unknown)
for video in test_data:
  start  = 24 
  if type(video) != str:
    video, start = video
  path = os.path.join(base_dir, video)
  annotate_video(path, False, start=start)
