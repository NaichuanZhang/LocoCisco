"""Keep the recording canvas's near-white surfaces distinct after H.264 export."""
import io

import av
import numpy as np

from unspoken.meeting.record_demo import screenshot_frame


def test_screenshot_colors_survive_video_encoding(tmp_path):
    pixels = np.zeros((64, 128, 3), dtype=np.uint8)
    pixels[:, :32] = (245, 246, 247)  # Canvas, distinct from the white cards.
    pixels[:, 32:64] = (255, 255, 255)
    pixels[:, 64:96] = (21, 32, 37)  # Dark text must retain detail.
    pixels[:, 96:] = (8, 101, 189)  # Blue accent.
    png = io.BytesIO()
    with av.open(png, "w", format="image2pipe") as image:
        stream = image.add_stream("png", rate=1)
        stream.width, stream.height, stream.pix_fmt = 128, 64, "rgb24"
        for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
            image.mux(packet)
        for packet in stream.encode():
            image.mux(packet)

    path = tmp_path / "canvas.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=1)
        stream.width, stream.height, stream.pix_fmt = 128, 64, "yuv420p"
        stream.codec_context.color_range = 1
        stream.codec_context.colorspace = 1
        frame = screenshot_frame(png.getvalue(), 128, 64)
        frame.pts = 0
        for packet in stream.encode(frame):
            output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    with av.open(str(path)) as encoded:
        decoded = next(encoded.decode(video=0)).to_ndarray(format="rgb24")
    for x in (16, 48, 80, 112):
        error = np.abs(decoded[32, x].astype(int) - pixels[32, x].astype(int))
        assert error.max() <= 4, (x, decoded[32, x], pixels[32, x])
