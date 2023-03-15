import sys

sys.path.append("/workspace/tensorrt/")
import vapoursynth as vs

core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4
core = vs.core
core.num_threads = 4  # can influence ram usage

core.std.LoadPlugin(path="/usr/lib/x86_64-linux-gnu/libffms2.so")
core.std.LoadPlugin(path="/usr/local/lib/libvstrt.so")
core.std.LoadPlugin(path="/usr/local/lib/libmvtools.so")

# cfr video
clip = core.ffms2.Source(source=globals()['video_path'], cache=False)
# vfr video (untested)
# clip = core.ffms2.Source(source='input.mkv', fpsnum = 24000, fpsden = 1001)

# resizing with descale
# Debilinear, Debicubic, Delanczos, Despline16, Despline36, Despline64, Descale
# clip = core.descale.Debilinear(clip, 1280, 720)

###############################################
# COLORSPACE
###############################################
# convert colorspace
clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s=globals()['matrix_in_s'])

for model in globals()['models'].split(','):
    clip = core.trt.Model(
       clip,
       engine_path=f"/workspace/tensorrt/models/{model}.engine",
       tilesize=[720, 540],
       num_streams=1,
    )    

###############################################
# OUTPUT
###############################################
clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

if eval(globals()['degrain']):
    super = core.mv.Super(clip)
    mvbw = core.mv.Analyse(super, isb=True, delta=1, overlap=4)    
    mvbw2 = core.mv.Analyse(super, isb=True, delta=2, overlap=4)
    mvbw3 = core.mv.Analyse(super, isb=True, delta=3, overlap=4)
    mvfw = core.mv.Analyse(super, isb=False, delta=1, overlap=4)
    mvfw2 = core.mv.Analyse(super, isb=False, delta=2, overlap=4)
    mvfw3 = core.mv.Analyse(super, isb=False, delta=3, overlap=4)
    clip = core.mv.Degrain3(clip=clip, super=super, mvbw=mvbw, mvfw=mvfw, mvbw2=mvbw2, mvfw2=mvfw2, mvbw3=mvbw3, mvfw3=mvfw3, thsad=400, thsadc=150)

clip.set_output()
