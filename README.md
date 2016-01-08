# 4dface: Real-time 3D face tracking and reconstruction from 2D video
[![Latest release](http://img.shields.io/github/release/patrikhuber/4dface.svg?style=flat-square)][release]
[![Linux build status of master branch](https://img.shields.io/travis/patrikhuber/4dface/master.svg?style=flat-square&label=Linux%3A%20build)][travis]
[![Windows build status of master branch](https://ci.appveyor.com/api/projects/status/ed5slvlhl0lpbu8j/branch/master?svg=true&passingText=Windows%3A%20build%20passing&failingText=Windows%3A%20build%20failing&pendingText=Windows%3A%20build%20pending)][appveyor]
[![Apache License 2.0](https://img.shields.io/github/license/patrikhuber/4dface.svg?style=flat-square)][license]
[![Webpage](https://img.shields.io/badge/webpage-www.4dface.org-blue.svg?style=flat-square)][webpage]

[release]: https://github.com/patrikhuber/4dface/releases
[travis]: https://travis-ci.org/patrikhuber/4dface
[appveyor]: https://ci.appveyor.com/project/patrikhuber/4dface/branch/master
[license]: https://github.com/patrikhuber/4dface/blob/master/LICENSE
[webpage]: http://www.4dface.org

This is a demo app showing face tracking and 3D Morphable Model fitting on live webcams and videos. It builds upon the 3D face model library [eos](https://github.com/patrikhuber/eos) and the landmark detection and optimisation library [superviseddescent](https://github.com/patrikhuber/superviseddescent).

(_caveat: due to recent additions, "real time" at the moment means around 5 fps._)

## Build & run

1. Clone with submodules: `git clone --recursive git://github.com/patrikhuber/4dface.git`, or, if you've already cloned it, get the submodules with `git submodule update --init --recursive` inside the `4dface` directory.

2. Make sure you've got boost (>=1.54.0 should do), OpenCV (>=2.4.8), Eigen (>=3.2.0) and a recent compiler (>=gcc-4.9, >=clang-3.6, >=VS2015) installed. For Ubuntu 14.04 and newer, this will do the trick:
    ```
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test
    sudo apt-get update
    sudo apt-get install gcc-5 g++-5 libboost-all-dev libeigen3-dev libopencv-dev opencv-data
    ```
    For Windows, get binaries for vc14-64bit (VS2015) from boost.org and opencv.org, and the Eigen headers.

3. Build the app:
    Run from _outside_ the source directory:
    1. `mkdir build && cd build`

    2. `cmake -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc-5 -DCMAKE_CXX_COMPILER=g++-5 -DOpenCV_haarcascades_DIR=/usr/share/opencv/haarcascades/ ../4dface/`

    On Windows, add `-G "Visual Studio 14 Win64"`. Also, you will probably need to add `-C ../4dface/initial_cache.cmake` as first argument - copy the file from `initial_cache.cmake.template` and adjust the paths.

    If you get an error about OpenCV\_haarcascades\_DIR, adjust `-DOpenCV_haarcascades_DIR` to point to the directory of `haarcascade_frontalface_alt2.xml` from OpenCV.

4. Type `make` or build in Visual Studio.

4. Type `make install`, or run the INSTALL target in Visual Studio, to copy all required files into a `share/` directory next to the executable.

Then just double-click the `4dface` app from the install-directory or run with `4dface -i videofile` to run on a video.

## Keyboard shortcuts

When running the 4dface app, `q` quits, `r` resets the tracking, and `s` saves an OBJ of the current model to the hard disk (into the directory where it was run from).

## Working with the libraries

If you're interested in working with the libraries, we recommend to clone and build them separately. They come with their own CMake project files and have their own GitHub issues pages.

* [eos](https://github.com/patrikhuber/eos): A lightweight header-only 3D Morphable Face Model fitting library in modern C++11/14
* [superviseddescent](https://github.com/patrikhuber/superviseddescent): A C++11 implementation of the supervised descent optimisation method

## License & contributions

This code is licensed under the Apache License, Version 2.0. The subprojects are also licensed under the Apache License, Version 2.0, except for the 3D morphable face model, which is free for use for non-commercial purposes - for commercial purposes, contact the [Centre for Vision, Speech and Signal Processing](http://www.surrey.ac.uk/cvssp/).

Contributions are very welcome! (best in the form of pull requests.) Please use Github issues for any bug reports, ideas, and discussions.

If you use this code in your own work, please cite one (or both) of the following papers:

* _Fitting 3D Morphable Models using Local Features_, P. Huber, Z. Feng, W. Christmas, J. Kittler, M. Rätsch, IEEE International Conference on Image Processing (ICIP) 2015, Québec City, Canada [[PDF]](http://arxiv.org/abs/1503.02330).

* _A Multiresolution 3D Morphable Face Model and Fitting Framework_, P. Huber, G. Hu, R. Tena, P. Mortazavian, W. Koppen, W. Christmas, M. Rätsch, J. Kittler, International Conference on Computer Vision Theory and Applications (VISAPP) 2016, Rome, Italy [[PDF]](http://www.patrikhuber.ch/files/3DMM_Framework_VISAPP_2016.pdf).
