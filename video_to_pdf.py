import os
import re
import whisper
import requests
import gdown
from fpdf import FPDF
from urllib.parse import urlparse
import yt_dlp

# ✅ Check if YouTube link
def is_youtube_url(url):
    return "youtube.com" in url or "youtu.be" in url

# ✅ Remove non-ASCII symbols (emojis etc.)
def clean_text(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# ✅ Transcribe video to text
def transcribe_video_to_text(video_path, model_size="base"):
    model = whisper.load_model(model_size)
    result = model.transcribe(video_path)
    return result["text"]

# ✅ Save transcript as PDF
def save_text_to_pdf(text, output_pdf_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    cleaned_text = clean_text(text)
    lines = cleaned_text.split('\n')

    for line in lines:
        pdf.multi_cell(0, 10, line)

    pdf.output(output_pdf_path)
    return output_pdf_path

# ✅ Convert video to PDF (transcribe and save)
def convert_video_to_pdf(video_path, output_pdf_path):
    transcript = transcribe_video_to_text(video_path)
    return save_text_to_pdf(transcript, output_pdf_path)

# ✅ Download direct video file (non-YouTube/Drive)
def download_direct_video(url, save_path):
    response = requests.get(url, stream=True)
    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("video"):
        raise Exception(f"Invalid content type: {content_type}. Not a video file.")
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return save_path

# ✅ Master function: YouTube, Google Drive, or direct link
def download_video_and_generate_pdf(video_url, save_dir="data"):
    os.makedirs(save_dir, exist_ok=True)

    if is_youtube_url(video_url):
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(video_url, download=False)
            title = info.get('title', 'video')
            video_id = info.get('id', '')
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(" ", "_")
        filename = f"{safe_title}_{video_id}.mp4"
        video_path = os.path.join(save_dir, filename)
        pdf_path = os.path.join(save_dir, f"{safe_title}_{video_id}.pdf")

        ydl_opts = {
            'format': 'mp4',
            'outtmpl': video_path,
            'quiet': True,
            'no_warnings': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

    elif "drive.google.com" in video_url:
        match = re.search(r'/d/([^/]+)', video_url)
        if not match:
            raise Exception("Invalid Google Drive link format.")
        file_id = match.group(1)
        direct_url = f"https://drive.google.com/uc?id={file_id}"
        filename = f"drive_video_{file_id}.mp4"
        video_path = os.path.join(save_dir, filename)
        pdf_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}.pdf")

        gdown.download(url=direct_url, output=video_path, quiet=False)

    else:
        parsed = urlparse(video_url)
        filename = os.path.basename(parsed.path) or "downloaded_video.mp4"
        filename = re.sub(r'[^\w\s-]', '', filename).strip().replace(" ", "_")
        video_path = os.path.join(save_dir, filename)
        pdf_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}.pdf")

        download_direct_video(video_url, video_path)

    return convert_video_to_pdf(video_path, pdf_path)

# ✅ Optional CLI mode
if __name__ == "__main__":
    url = input("Enter YouTube, Google Drive, or direct video link: ")
    try:
        pdf = download_video_and_generate_pdf(url)
        print(f"\n✅ Transcript saved at: {pdf}")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
