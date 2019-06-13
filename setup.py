from setuptools import setup, find_packages

requirements = [
    'numpy',
    'opencv-python'
]

dev_requirements = [
    'pip-tools',
]

setup(
    name='piano_vision',
    version='0.0.1',
    description='Automatic transcription and assisted tutoring for amateur piano players.',
    author='Alex Tompkins',
    author_email='alex.tompkins@atomic.nz',
    url='https://github.com/alextompkins/piano-vision',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        'dev': dev_requirements
    }
)
