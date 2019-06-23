from piano_vision.main import PianoVision


VIDEO_NAME = 'canon_in_d'


if __name__ == '__main__':
	piano_vision = PianoVision(VIDEO_NAME)
	piano_vision.main_loop()
