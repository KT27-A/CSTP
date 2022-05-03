# modified from https://github.com/craston/MARS
'''
Kinetics: Minimum resolution 320, FPS 30
UCF-101/HMDB-51: Minimum resolution 256, FPS 25
'''
import sys, os, pdb
import numpy as np
import subprocess
from tqdm import tqdm


def extract(vid_dir, frame_dir, start, end, redo=False, res=320, fps=30):
  class_list = sorted(os.listdir(vid_dir))[start:end]
  print("Classes =", class_list)
  for ic, cls in enumerate(class_list):
    vlist = sorted(os.listdir(vid_dir + cls))
    print("")
    print(ic+1, len(class_list), cls, len(vlist))
    print("")
    for v in tqdm(vlist):
      outdir = os.path.join(frame_dir, cls, v[:-4])

      # Checking if frames already extracted
      if os.path.isfile(os.path.join(outdir, 'done')) and not redo: continue
      try:
        os.system('mkdir -p "%s"' % (outdir))
        # check if horizontal or vertical scaling factor
        o = subprocess.check_output('ffprobe -v error -show_entries stream=width,height -of default=noprint_wrappers=1 "%s"'%(os.path.join(vid_dir, cls, v)), shell=True).decode('utf-8')
        lines = o.splitlines()
        width = int(lines[0].split('=')[1])
        height = int(lines[1].split('=')[1])
        resize_str = '-1:{}'.format(res) if width > height else '{}:-1'.format(res)

        # extract frames
        os.system('ffmpeg -i "%s" -r "%s" -q:v 2 -vf "scale=%s" "%s"  > /dev/null 2>&1'%(os.path.join(vid_dir, cls, v), str(fps), resize_str, os.path.join(outdir, '%05d.jpg')))
        nframes = len([fname for fname in os.listdir(outdir) if fname.endswith('.jpg') and len(fname) == 9])
        if nframes==0: raise Exception

        os.system('touch "%s"' % (os.path.join(outdir, 'done')))
      except:
        print("ERROR", cls, v)


if __name__ == '__main__':
  vid_dir = './kinetics_400_mmlab/kinetics_400_train/'
  frame_dir = './kinetics_400_mmlab_1f_320/kinetics_400_train/'
  start = 0
  end = 400
  res = 320  # resolution. kinetics: 320; UCF-101/HMDB-51: 256
  fps = 30  # FPS. Kinetics: 30; UCF-101/HMDB-51: 25

  extract(vid_dir, frame_dir, start, end, redo=True)
