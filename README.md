This is a project where I do my CV research and development for projects before they make it into those projects. It contains various test and utility scripts.

The calibration folder contains a script for obtaining the intrinsic and extrinsic parameters of a camera as well as its distortion coefficients. It is also able to undistort the images it works with.

The pipeline_configurator folder contains some desktop applications for interacting with pipelines in real-time. The tkinter version should be used as the OpenCV version is very out-of-date.

The pipelines folder contains the implmentations for various pipelines, such as the FRC pipeline.

The pnp folder has some pnp test scripts, including one that can be used with a Limelight, and misc folder contains some miscellaneous test scripts.

Make sure you download the Common submodule, which contains a lot of the code for the newer implementations of the pipeline. You can install it by navigating to the root of the project in your terminal and running "pip install -e Common". And it's always good practice to use a virtual environment.

More documentation will come soon!