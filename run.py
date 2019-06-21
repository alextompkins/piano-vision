from piano_vision.main import PianoVision


VIDEO_NAME = 'call_me_maybe'


if __name__ == '__main__':
	piano_vision = PianoVision(VIDEO_NAME)
	piano_vision.main_loop()
