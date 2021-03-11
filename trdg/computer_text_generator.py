import random as rnd
from PIL import Image, ImageColor, ImageFont, ImageDraw, ImageFilter
import numpy as np
import cv2

def generate(
    text,
    font,
    text_color,
    font_size,
    orientation,
    space_width,
    character_spacing,
    fit,
    word_split,
    stroke_width=0, 
    stroke_fill="#282828",
):
    if orientation == 0:
        return _generate_horizontal_text(
            text,
            font,
            text_color,
            font_size,
            space_width,
            character_spacing,
            fit,
            word_split,
            stroke_width,
            stroke_fill,
        )
    elif orientation == 1:
        return _generate_vertical_text(
            text, font, text_color, font_size, space_width, character_spacing, fit,
            stroke_width, stroke_fill, 
        )
    else:
        raise ValueError("Unknown orientation " + str(orientation))


def _generate_horizontal_text(
    text, font, text_color, font_size, space_width, character_spacing, fit, word_split, 
    stroke_width=0, stroke_fill="#282828"
):
    image_font = ImageFont.truetype(font=font, size=font_size)

    space_width = int(image_font.getsize(" ")[0] * space_width)

    if word_split:
        splitted_text = []
        for w in text.split(" "):
            splitted_text.append(w)
            splitted_text.append(" ")
        splitted_text.pop()
    else:
        splitted_text = text

    piece_widths = [
        image_font.getsize(p)[0] if p != " " else space_width for p in splitted_text
    ]
    text_width = sum(piece_widths)
    if not word_split:
        text_width += character_spacing * (len(text) - 1)

    text_height = max([image_font.getsize(p)[1] for p in splitted_text])

    txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    txt_mask = Image.new("RGB", (text_width, text_height), (0, 0, 0))

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask, mode="RGB")
    txt_mask_draw.fontmode = "1"

    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    c1, c2 = colors[0], colors[-1]

    fill = (
        rnd.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
        rnd.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
        rnd.randint(min(c1[2], c2[2]), max(c1[2], c2[2])),
    )

    stroke_colors = [ImageColor.getrgb(c) for c in stroke_fill.split(",")]
    stroke_c1, stroke_c2 = stroke_colors[0], stroke_colors[-1]

    stroke_fill = (
        rnd.randint(min(stroke_c1[0], stroke_c2[0]), max(stroke_c1[0], stroke_c2[0])),
        rnd.randint(min(stroke_c1[1], stroke_c2[1]), max(stroke_c1[1], stroke_c2[1])),
        rnd.randint(min(stroke_c1[2], stroke_c2[2]), max(stroke_c1[2], stroke_c2[2])),
    )

    for i, p in enumerate(splitted_text):
        txt_img_draw.text(
            (sum(piece_widths[0:i]) + i * character_spacing * int(not word_split), 0),
            p,
            fill=fill,
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        txt_mask_draw.text(
            (sum(piece_widths[0:i]) + i * character_spacing * int(not word_split), 0),
            p,
            fill=((i + 1) // (255 * 255), (i + 1) // 255, (i + 1) % 255),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )

    if fit:
        return txt_img.crop(txt_img.getbbox()), txt_mask.crop(txt_img.getbbox())
    else:
        return txt_img, txt_mask


def _generate_vertical_text(
    text, font, text_color, font_size, space_width, character_spacing, fit,
    stroke_width=0, stroke_fill="#282828"
):
    image_font = ImageFont.truetype(font=font, size=font_size)

    space_height = int(image_font.getsize(" ")[1] * space_width)

    char_heights = [
        image_font.getsize(c)[1] if c != " " else space_height for c in text
    ]
    text_width = max([image_font.getsize(c)[0] for c in text])
    text_height = sum(char_heights) + character_spacing * len(text)

    txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    txt_mask = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask)

    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    c1, c2 = colors[0], colors[-1]

    fill = (
        rnd.randint(c1[0], c2[0]),
        rnd.randint(c1[1], c2[1]),
        rnd.randint(c1[2], c2[2]),
    )

    stroke_colors = [ImageColor.getrgb(c) for c in stroke_fill.split(",")]
    stroke_c1, stroke_c2 = stroke_colors[0], stroke_colors[-1] 

    stroke_fill = (
        rnd.randint(stroke_c1[0], stroke_c2[0]),
        rnd.randint(stroke_c1[1], stroke_c2[1]),
        rnd.randint(stroke_c1[2], stroke_c2[2]),
    )

    for i, c in enumerate(text):
        txt_img_draw.text(
            (0, sum(char_heights[0:i]) + i * character_spacing),
            c,
            fill=fill,
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        txt_mask_draw.text(
            (0, sum(char_heights[0:i]) + i * character_spacing),
            c,
            fill=(i // (255 * 255), i // 255, i % 255),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )

    if fit:
        return txt_img.crop(txt_img.getbbox()), txt_mask.crop(txt_img.getbbox())
    else:
        return txt_img, txt_mask

def plain_white(height, width):
    """
        Create a plain white background
    """

    return Image.new("L", (width, height), 255).convert("RGBA")

def judge_normal(text, font):
    normal = True
    font_size = 32
    text_color = "#282828"
    space_width = 1.0
    character_spacing = 0
    fit = False
    word_split = False
    text_img, _ = _generate_horizontal_text(
        text, font, text_color, font_size, space_width, character_spacing, fit, word_split,
        stroke_width=0, stroke_fill="#282828")
    w = text_img.size[0] + 10
    h = text_img.size[1] + 10
    bg = plain_white(h, w)
    bg.paste(text_img, (5, 5), text_img)
    text_img = np.array(bg)
    gray = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 1:
        normal = False
    else:
        if len(contours) == 3:
            if len(contours[1]) == 8 and len(contours[2]) == 4:
                normal = False
    return normal






if __name__ == '__main__':
    import numpy as np
    import cv2
    import os
    from tqdm import tqdm
    lines = open(r"C:\DM_hcn\project\data_generator\TextRecognitionDataGenerator\trdg\dicts\char_chinese_8637.txt", "r", encoding="utf-8").read().strip().strip("\n").split("\n")
    char_list = []
    save_dir = "out4"
    normal_dir = os.path.join(save_dir, "normal")
    un_normal_dir = os.path.join(save_dir, "un_normal")
    if not os.path.exists(normal_dir):
        os.mkdir(normal_dir)
    if not os.path.exists(un_normal_dir):
        os.mkdir(un_normal_dir)
    for i, line in tqdm(enumerate(lines)):
        char_list.append(line[0])
        text = line[0]
        # text = "ɔ"
        # text = "你"
        font = r"C:\DM_hcn\project\data_generator\TextRecognitionDataGenerator\trdg\fonts\cn\STKAITI.TTF"
        font_size = 32
        text_color = "#282828"
        space_width = 1.0
        character_spacing = 0
        fit = False
        word_split = False
        text_img, _ = _generate_horizontal_text(
            text, font, text_color, font_size, space_width, character_spacing, fit, word_split,
            stroke_width=0, stroke_fill="#282828"
        )
        w = text_img.size[0] + 10
        h = text_img.size[1] + 10
        bg = plain_white(h, w)
        bg.paste(text_img, (5, 5), text_img)
        text_img = np.array(bg)
        gray = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if judge_normal(text, font):
            bg.save(os.path.join(normal_dir, "test_" + str(i) + ".png"))
        else:
            bg.save(os.path.join(un_normal_dir, "test_" + str(i) + ".png"))
        # if len(contours) == 1:
        #     # cv2.imwrite("gray.png", gray)
        #     # cv2.imwrite("binary.png", binary)
        #     bg.save(os.path.join(un_normal_dir, "test_" + str(i) + ".png"))
        # else:
        #     if len(contours)==3:
        #         if len(contours[1]) == 8 and len(contours[2]) == 4:
        #             bg.save(os.path.join(un_normal_dir, "test_" + str(i) + ".png"))
        #         else:
        #             bg.save(os.path.join(normal_dir, "test_" + str(i) + ".png"))
        #     else:
        #         bg.save(os.path.join(normal_dir, "test_" + str(i) + ".png"))
        # for contour in contours:
        #     print(contour)
    labelfile = open(os.path.join(save_dir, "label.txt"), "w", encoding="utf-8")
    for i, char in enumerate(char_list):
        labelfile.write(os.path.join(un_normal_dir, "test_" + str(i) + ".png") + " " + char + "\n")


