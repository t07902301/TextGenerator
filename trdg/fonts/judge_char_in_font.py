import os
import glob
import time
from fontTools.ttLib import TTFont
from fontTools.ttLib import TTCollection
pwd = "cn"
fontpaths = glob.glob(os.path.join(pwd, "*.ttc"))
# fonts = []
fontpaths.extend(glob.glob(os.path.join(pwd, "*.ttf")))
fontpaths.extend(glob.glob(os.path.join(pwd, "*.TTF")))


def judge_char_in_font1(ch, font_path, fonts):
    # if font_path.endswith("ttc"):
    #     fonts = TTCollection(font_path)
    # else:
    #     fonts = [TTFont(font_path)]
    found = False
    for font in fonts:
        for table in font['cmap'].tables:
            if found:
                break
            if ord(ch) in table.cmap:
                found = True
                break
    if not found:
        print(ch)
        print("{}不在字体{}内".format(ch, os.path.basename(font_path)))
    return found


def judge_char_in_font2(ch, font_path, fonts):
    # if font_path.endswith("ttc"):
    #     fonts = TTCollection(font_path)
    # else:
    #     fonts = [TTFont(font_path)]
    glyph_name = None
    for font in fonts:
        for table in font['cmap'].tables:
            if glyph_name is not None:
                break
            glyph_name = table.cmap.get(ord(ch))

    if glyph_name is not None:
        glyf = font['glyf']
        found = glyf.has_key(glyph_name) and glyf[glyph_name].numberOfContours > 0
    else:
        found = False
    if not found:
        print("{}不在字体{}内".format(ch, os.path.basename(font_path)))
    return found


t1 = time.time()
font_dict = {}
for font_path in fontpaths:
    if font_path.endswith("ttc"):
        font_dict[font_path] = TTCollection(font_path)
    else:
        font_dict[font_path] = [TTFont(font_path)]
for i in range(1000):
    for font_path in fontpaths:
        judge_char_in_font1("㓱", font_path, font_dict[font_path])
        judge_char_in_font2("㓱", font_path, font_dict[font_path])
t2 = time.time()
print(t2 - t1)


