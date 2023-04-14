import click
import subprocess

@click.command()
@click.option("--preprocessed-file", required=True, type=str, help="The preprocessed file.")
@click.option("--models", required=True, type=str, help="A comma-separated list of upscale model names.")
@click.option("--color-matrix", required=True, type=str, help="The input color matrix.")
@click.option("--degrain/--no-degrain", required=True, help="Apply an MDegrain3 filter.")
@click.option("--encode/--no-encode", required=True, help="Encode the upscale.")
@click.option("--crf", required=True, type=float, help="The encode CRF.")
@click.option("--x265-params", required=True, type=str, help="The x265 parameters.")
@click.option("--output-file", required=True, type=str, help="The output file.")

def upscale(preprocessed_file: str, models: str, color_matrix: str, degrain: bool, encode: bool, crf: float, x265_params: str, output_file: str):
    vspipe_arguments = [
        "vspipe",
        "-c", "y4m",
        "/workspace/tensorrt/upscale/vapoursynth.py",
        "-a", f"video_path={preprocessed_file}",
        "-a", f"models={models}",
        "-a", f"matrix_in_s={color_matrix}",
        "-a", f"degrain={degrain}",
        # write to standard output
        "-"
    ]

    ffmpeg_arguments = [
        "ffmpeg",
        "-i", "pipe:",
        "-i", preprocessed_file,
        "-s", "1440:1080",
        "-sws_flags", "bicubic",
        "-map", "0:v",
        "-map", "1:a",
        "-map", "1:s?",
        "-c", "copy",
        "-c:v", "libx265" if encode else "libx264",
        "-preset", "slow" if encode else "ultrafast",
        "-crf", str(crf if encode else 0),
        "-pix_fmt", "yuv420p10le",
        *(["-x265-params", x265_params] if encode else []),
        output_file
    ]

    print(f"{' '.join(vspipe_arguments)} | {' '.join(ffmpeg_arguments)}")

    vspipe = subprocess.Popen(vspipe_arguments, stdout=subprocess.PIPE)
    ffmpeg = subprocess.Popen(ffmpeg_arguments, stdin=vspipe.stdout)
    ffmpeg.communicate()

if __name__ == "__main__":
    upscale()