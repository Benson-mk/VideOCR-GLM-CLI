# VideOCR-GLM-CLI

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Extract hardcoded subtitles from videos!

## ℹ About

Extract hardcoded (burned-in) subtitles from videos via command line by utilizing the [GLM-OCR](https://github.com/zai-org/GLM-OCR) OCR engine through the Ollama API. Everything can be easily configured via command-line parameters.

This project is inspired by [timminator/VideOCR](https://github.com/timminator/VideOCR) but replaces PaddleOCR with GLM-OCR for improved text recognition capabilities.

## Setup

### Prerequisites

- Python 3.9 or higher
- [Ollama](https://ollama.ai/) installed and running
- GLM-OCR model pulled in Ollama

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Benson-mk/VideOCR-GLM-CLI.git
cd VideOCR-GLM-CLI
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install and setup Ollama:
```bash
# Install Ollama (if not already installed)
# Visit https://ollama.ai/ for installation instructions

# Pull the GLM-OCR model
ollama pull glm-ocr:latest

# Start Ollama server (if not running)
ollama serve
```

## Usage

Run the script from the command line with the required parameters:

```bash
python videocr_glm_cli.py --video_path "path/to/video.mp4" --output "output.srt"
```

### Example usage:

```bash
python videocr_glm_cli.py --video_path "Path\to\your\video\example.mp4" --output "Path\to\your\desired\subtitle\location\example.srt" --lang en --time_start "18:40"
```

For a complete list of available parameters, run:

```bash
python videocr_glm_cli.py --help
```

## Performance

The OCR process speed depends on your Ollama setup and hardware. Using a GPU-accelerated Ollama instance is recommended for better performance. The GLM-OCR model provides excellent text recognition accuracy, especially for mixed-language content.

## Tips

When cropping, leave a bit of buffer space above and below the text to ensure accurate readings, but also don't make the box too large.

### Quick Configuration Cheatsheet

| | More Speed | More Accuracy | Notes |
|---|------------|---------------|-------|
| Input Video Quality | Use lower quality | Use higher quality | Performance impact of using higher resolution video can be reduced with cropping |
| `frames_to_skip` | Higher number | Lower number | For perfectly accurate timestamps this parameter needs to be set to 0. |
| `SSIM threshold` | Lower threshold | Higher Threshold | If the SSIM between consecutive frames exceeds this threshold, the frame is considered similar and skipped for OCR. A lower value can greatly reduce the number of images OCR needs to be performed on. |

## Command Line Parameters

### Video Parameters

- `--video_path` (required)
  
  Path for the video where subtitles should be extracted from.

- `--output` (default: `subtitle.srt`)
  
  Path for the desired location where the .srt file should be stored.

- `--time_start` (default: `0:00`)
  
  Start time for subtitle extraction in MM:SS or HH:MM:SS format.

- `--time_end` (default: empty)
  
  End time for subtitle extraction in MM:SS or HH:MM:SS format. Extract subtitles from only a clip of the video. The subtitle timestamps are still calculated according to the full video length.

### OCR Parameters

- `--lang` (default: `en`)
  
  The language of the subtitles. Supports a wide range of languages including English, Chinese (Simplified/Traditional), Japanese, Korean, Arabic, Cyrillic, and many more.

- `--subtitle_position` (default: `center`)
  
  Specifies the alignment of subtitles in the video and allows for better text recognition. Options: `center`, `left`, `right`, `any`.

- `--sim_threshold` (default: `80`)
  
  Similarity threshold for subtitle lines. Subtitle lines with larger [Levenshtein](https://en.wikipedia.org/wiki/Levenshtein_distance) ratios than this threshold will be merged together. The default value `80` is fine for most cases.
  
  Make it closer to 0 if you get too many duplicated subtitle lines, or make it closer to 100 if you get too few subtitle lines.

- `--ssim_threshold` (default: `92`)
  
  If the SSIM between consecutive frames exceeds this threshold, the frame is considered similar and skipped for OCR. A lower value can greatly reduce the number of images OCR needs to be performed on. On relatively tight crop boxes around the subtitle area good results could be seen with this value all the way lowered to 85.

- `--post_processing` (default: `false`)
  
  This parameter adds a post processing step to the subtitle detection. The detected text will be analyzed for missing spaces and tries to insert them automatically.

- `--max_merge_gap` (default: `0.09`)
  
  Maximum allowed time gap (in seconds) between two subtitles to be considered for merging if they are similar. The default value 0.09 (i.e., 90 milliseconds) works well in most scenarios.
  
  Increase this value if you notice that the output SRT file contains several subtitles with the same text that should be merged into a single one and are wrongly split into multiple ones.

- `--use_fullframe` (default: `false`)
  
  By default, the specified cropped area is used for OCR or if a crop is not specified, then the bottom third of the frame will be used. By setting this value to `true` the entire frame will be used.

- `--use_dual_zone` (default: `false`)
  
  This parameter allows to specify two areas that will be used for OCR.

- `--crop_x`, `--crop_y`, `--crop_width`, `--crop_height`
  
  Specifies the bounding area in pixels for the portion of the frame that will be used for OCR (Zone 1).

- `--crop_x2`, `--crop_y2`, `--crop_width2`, `--crop_height2`
  
  Specifies the bounding area in pixels for the second OCR zone (Zone 2). Only used when `--use_dual_zone` is enabled.

- `--ocr_image_max_width` (default: `960`)
  
  Downscales the cropped image frame so its width does not exceed this value before passing it to the OCR engine. A lower value shortens the processing time, but setting it too low can reduce OCR accuracy.

- `--brightness_threshold` (default: `None`)
  
  If set, pixels whose brightness are less than the threshold will be blackened out. Valid brightness values range from 0 (black) to 255 (white). This can help improve accuracy when performing OCR on videos with white subtitles.

- `--frames_to_skip` (default: `1`)
  
  The number of frames to skip before sampling a frame for OCR. Keep in mind the fps of the input video before increasing.

- `--min_subtitle_duration` (default: `0.2`)
  
  Subtitles shorter than this threshold (in seconds) will be omitted from the final subtitle file.

### Ollama Parameters

- `--ollama_host` (default: `localhost`)
  
  Ollama server host address.

- `--ollama_port` (default: `11434`)
  
  Ollama server port.

- `--ollama_model` (default: `glm-ocr:latest`)
  
  Ollama model name to use for OCR.

- `--ollama_timeout` (default: `300`)
  
  Ollama API timeout in seconds.

- `--allow_system_sleep` (default: `false`)
  
  Allow the system to sleep during processing. When set to `false`, the system will be kept awake during the entire OCR process.

## Acknowledgments

This project is inspired by and based on the excellent work of:

- [timminator/VideOCR](https://github.com/timminator/VideOCR) - Original VideOCR project structure and implementation
- [zai-org/GLM-OCR](https://github.com/zai-org/GLM-OCR) - GLM-OCR model for text recognition

## License

This project is licensed under the MIT License - see the LICENSE file for details.