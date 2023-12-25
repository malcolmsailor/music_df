import os
from types import MappingProxyType

from PIL import Image, ImageDraw, ImageFont

# From an email from Craig Sapp
# USER_SIGNIFIERS = set("@NZ|+ij")
# DEFAULT_COLOR_MAPPING = MappingProxyType(
#     {
#         # "@": "#007bff",
#         # "N": "#6610f2",
#         # "Z": "#e83e8c",
#         # "|": "#fd7e14",
#         # "+": "#28a745",
#         # "i": "#17a2b8",
#         # "j": "#343a40",
#         "@": "#FF0000",
#         "N": "#0000FF",
#         "Z": "#00FF00",
#         "|": "#800080",
#         "+": "#FF00FF",
#         "i": "#00FFFF",
#         "j": "#FFa500",
#     }
# )

HEX_CODES = """
#FF0000
#00FF00
#0000FF
#FFFF00
#FF00FF
#00FFFF
#800000
#008000
#000080
#808000
#800080
#008080
#FF8080
#80FF80
#8080FF
#FFFF80
#FF80FF
#80FFFF
#c08080
#80c080
#8080c0
#c0c080
#c080c0
#80c0c0
#FF4040
#40FF40
#4040FF
#FFFF40
#FF40FF
#40FFFF
#c00000
#00c000
#0000c0
#c0c000
#c000c0
#00c0c0
""".strip().split()

EMOJI = [chr(x) for x in range(0x1F600, 0x1F650)]
DINGBATS = [chr(x) for x in range(0x2700, 0x27C0)]
MISC_SYMBOLS = [chr(x) for x in range(0x2600, 0x2700)]

USER_SIGNIFIERS = EMOJI + DINGBATS + MISC_SYMBOLS

DEFAULT_COLOR_MAPPING = dict(zip(USER_SIGNIFIERS, HEX_CODES))


def preview_colors(
    hex_codes: list[str],
    output_path: str,
    rect_width: int = 40,
    font_size: int = 20,
    padding: int = 10,
    width: int = 600,
    # font_path: str = "/Library/Fonts/Arial Unicode.ttf",
):
    def get_height(data):
        return len(data) * (font_size + padding) + padding

    img = Image.new("RGB", (width, get_height(hex_codes)), "white")
    d = ImageDraw.Draw(img)
    # font = ImageFont.truetype(font_path, font_size)

    y_position = padding
    for hex_code in hex_codes:
        d.rectangle(
            (
                padding,
                y_position,
                padding + rect_width,
                y_position + font_size,
            ),
            fill=hex_code,
        )
        d.text(
            (padding + rect_width + padding, y_position),
            hex_code,
            # font=font,
            fill="black",
        )
        y_position += font_size + padding

    img.save(output_path)


if __name__ == "__main__":
    preview_colors(HEX_CODES, os.path.expanduser("~/tmp/hex_codes.png"))  # type:ignore
