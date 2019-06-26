import sys
from piano_vision.main import PianoVision


VIDEO_NAME = 'canon_in_d'


if __name__ == '__main__':
	video_name = VIDEO_NAME

	if len(sys.argv) >= 2:
		video_name = sys.argv[1]

	piano_vision = PianoVision(video_name)
	piano_vision.main_loop()
