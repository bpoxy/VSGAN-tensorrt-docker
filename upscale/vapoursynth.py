import sys

sys.path.append("/workspace/tensorrt/")
import vapoursynth as vs

core = vs.core
core.num_threads = 4  # can influence ram usage

core.std.LoadPlugin(path="/usr/local/lib/libvstrt.so")
clip = core.bs.VideoSource(source=globals()['video_path'])

###############################################
# COLORSPACE
###############################################
clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s=globals()['matrix_in_s'])

###############################################
# UPSCALE
###############################################
for model in globals()['models'].split(','):
    clip = core.trt.Model(
       clip,
       engine_path=f"/workspace/tensorrt/models/{model}.engine",
       tilesize=[clip.width, clip.height],
       num_streams=1,
    )    

###############################################
# OUTPUT
###############################################
clip = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
clip.set_output()
