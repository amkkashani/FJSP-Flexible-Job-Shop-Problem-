import os
import re
from PIL import Image

def frames_to_gif(
    input_dir=".",
    output_path="output.gif",
    fps=15,
    pattern=r"^frame_(\d+)\.(png|jpg|jpeg|bmp|webp)$",
):
    rx = re.compile(pattern, re.IGNORECASE)

    frames = []
    for name in os.listdir(input_dir):
        m = rx.match(name)
        if m:
            frames.append((int(m.group(1)), name))

    if not frames:
        raise FileNotFoundError(f"No frames found in '{input_dir}' matching {pattern}")

    frames.sort(key=lambda x: x[0])

    images = []
    base_size = None

    for _, fname in frames:
        fpath = os.path.join(input_dir, fname)
        img = Image.open(fpath).convert("RGBA")

        if base_size is None:
            base_size = img.size
        elif img.size != base_size:
            img = img.resize(base_size, Image.Resampling.LANCZOS)

        images.append(img)

    duration_ms = int(1000 / fps)  # per frame
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"Done. Wrote {len(images)} frames to {output_path}")

if __name__ == "__main__":
    frames_to_gif(
        input_dir=".",
        output_path="output.gif",
        fps=15,
    )
