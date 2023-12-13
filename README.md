# Conv-fusion Transformer with Distribution Contrast Loss
The official implementation of "Conv-fusion Transformer with Distribution Contrast Loss for Surface Depth Estimation from Multi-view Stereo Satellite Images”
<br>
The experimental results are available here, and the full code will be made publicly available upon acceptance. The weights files are available <a href="https://github.com/ZYangChen/CTD-Sat/releases/tag/checkpints">here</a>.
## Environment preparation
```Shell
conda create -n sat1 python=3.7
conda activate sat1
wget -c https://www.sqlite.org/2021/sqlite-autoconf-3340100.tar.gz
tar -xvf sqlite-autoconf-3340100.tar.gz
cd sqlite-autoconf-3340100
vim sqlite3.c
```
Add macros under "include"
```Shell
#define SQLITE_CORE 1
#define SQLITE_AMALGAMATION 1
#ifndef SQLITE_PRIVATE
# define SQLITE_PRIVATE static
#endif
#define SQLITE_ENABLE_COLUMN_METADATA 1        //Please pay attention to this line
 
/************** Begin file ctime.c *******************************************/
/*
```
recompile
```Shell
./configure
make
sudo make uninstall
sudo make install
#Verify that the installation was successful
sqlite3 --version
```
Download proj (version 6.3.2) source code, unzip it and compile and install it
```Shell
proj-5.2.0
wget https://download.osgeo.org/proj/proj-6.3.2.tar.gz
tar -zxvf proj-6.3.2.tar.gz
#Go to the directory and compile
cd proj-6.3.2
./configure
make
make install
ldconfig
proj --version
```
Download geos (version 3.8.1), unzip it, compile and install it.
```Shell
wget http://download.osgeo.org/geos/geos-3.8.1.tar.bz2
tar -jxvf geos-3.8.1.tar.bz2
cd geos-3.8.1
./configure
make
make install 
ldconfig
geos-config --version
```
gdal(2.4.2)
```Shell
pip install setuptools==57.5.0
sudo add-apt-repository ppa:ubuntugis && sudo apt update
sudo apt install gdal-bin
gdalinfo --version  # 假设输出为2.4.2
```
```Shell
pip install gdal==2.4.2.*
```
or
```Shell
wget -c http://download.osgeo.org/gdal/2.4.2/gdal-2.4.2.tar.gz
tar -zxvf gdal-2.4.2.tar.gz
cd /gdal-2.4.2/swig/python/
python setup.py build
python setup.py install
```
```Shell
python
from osgeo import gdal
```
```Shell
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorboardX
pip install matplotlib
pip install opencv-python
pip install imageio
```
