import argparse
import re
import sys
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont


@dataclass
class Settings:
    font_size: int = 20
    padding: int = 10
    rect_width: int = 40
    width: int = 800
    font_path: str = "/Library/Fonts/Arial Unicode.ttf"

    def get_height(self, data):
        return (len(data) + 2) * (self.font_size + self.padding) + self.padding


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("humdrum_file")
    parser.add_argument("output_path")
    args = parser.parse_args()
    return args


def main(humdrum_path, output_path, settings=Settings()):
    with open(humdrum_path) as inf:
        data = inf.read()
    # If colors have transparency values, then they will be 8 characters long, but
    #   we want to ignore the transparencies.
    matches = re.findall(r"!! color_key: ([^=]+)=(#[a-fA-F0-9]{6})", data)
    if not matches:
        print(f"No colors found!")
        sys.exit(1)

    img = Image.new("RGB", (settings.width, settings.get_height(matches)), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(settings.font_path, settings.font_size)

    y_position = settings.padding
    for key, color in matches:
        d.rectangle(
            (
                settings.padding,
                y_position,
                settings.padding + settings.rect_width,
                y_position + settings.font_size,
            ),
            fill=color,
        )
        d.text(
            (settings.padding + settings.rect_width + settings.padding, y_position),
            key,
            font=font,
            fill="black",
        )
        y_position += settings.font_size + settings.padding

    d.text(
        (settings.padding, y_position),
        "Notes are colored according to labels (correct/incorrect)",
        font=font,
        fill="black",
    )
    d.text(
        (settings.padding, y_position + settings.font_size + settings.padding),
        "Red text annotations show model's incorrect predictions",
        font=font,
        fill="black",
    )

    img.save(output_path)


if __name__ == "__main__":
    args = parse_args()
    main(args.humdrum_file, args.output_path)
