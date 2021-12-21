import pytube
from pytube.cli import on_progress

url = "유튜브url"  # https://www.youtube.com/watch?v=sMcvpUxe2F0
yt = pytube.YouTube(url, on_progress_callback=on_progress)
save_dir = "./"  # 저장경로

yt.streams.filter(progressive=True, file_extension="mp4")\
    .order_by("resolution")\
    .desc()\
    .first()\
    .download(save_dir)

