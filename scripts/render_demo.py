"""Render captions below the original recording without covering product evidence."""
import argparse
from pathlib import Path
import re
import subprocess


def render(raw,subtitles,output):
    subtitles,output=Path(subtitles),Path(output)
    ass=output.with_suffix('.ass')
    events=[]
    def stamp(value):
        hours,minutes,seconds,milliseconds=map(int,re.split('[:,]',value))
        return f'{hours}:{minutes:02}:{seconds:02}.{milliseconds//10:02}'
    for block in subtitles.read_text().strip().split('\n\n'):
        lines=block.splitlines()
        start,end=lines[1].split(' --> ')
        caption=r'\N'.join(lines[2:])
        events.append(f'Dialogue: 0,{stamp(start)},{stamp(end)},Default,,0,0,0,,{caption}')
    ass.write_text('''[Script Info]
ScriptType: v4.00+
PlayResX: 1440
PlayResY: 1140
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,26,&H00FFFFFF,&H00FFFFFF,&H0015283F,&H0015283F,0,0,0,0,100,100,0,0,1,0,0,2,40,40,38,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''+ '\n'.join(events)+'\n')
    # Working in the output directory keeps the filter path simple and portable.
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(Path(raw).resolve()),
        '-vf',f'pad=iw:ih+140:0:0:color=0x15283f,ass={ass.name}',
        '-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',output.name],
        cwd=output.resolve().parent,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--raw',required=True)
    p.add_argument('--subtitles',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    render(a.raw,a.subtitles,a.output)
