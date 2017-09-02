### Setting up environment on Ubuntu

#### 1. Install latest NVIDIA gpu driver.   
    
More often than not, third party driver is more up-to-date than official Ubuntu release for
nvidia driver. To use a third party driver, first disable secure boot. Then :    
    `$ sudo -i`    
    `$ add-apt-repository ppa:graphics-drivers/ppa`    
    `$ apt-get update`    
    `$ apt-get install nvidia-381`    
    `$ shutdown -r now`    
        
#### 2. Install CUDA toolkit.    
    
Follow the instructions here: http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation    
Also read this blog post before installing : https://blog.nelsonliu.me/2017/04/29/installing-and-updating-gtx-1080-ti-cuda-drivers-on-ubuntu/    
Use runfile ( download from NVIDIA website ) to install so you can opt out of installing the driver that comes with cuda toolkit.   
        
#### 3. Install cuDNN.    
    
Download and install cuDNN .deb files from NVIDIA developer site : https://developer.nvidia.com/rdp/cudnn-download    
Install runtime first, then developer package.    
Some version of Tensorflow depends on cuDNN version 5. Tensorflow will yield error message if you don't have this.    
For example, Tensorflow 1.2.0 depends on cuDNN 5.1.    
Download cuDNN 5.1 files and copy them to `/usr/local/cuda/lib64/` and add this line in `~/.bashrc`:    
> export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"    
        
#### 4. Download and install Anaconda.    
    
Don't use sudo privilege when running the installation script. 
    
#### 5. Create a new environment and install tensorflow.

First create a new conda environment:    
`$ conda create -n tf_1_2 python=3.5`    
    
Activate environment:    
`$ source activate tf_1_2`
    
Install tensorflow:    
<pre>(tf_1_2) $ pip install --ignore-installed --upgrade <i>tfbinaryURL</i></pre>
    
*tfbinaryURL* can be found here: https://www.tensorflow.org/install/install_linux#the_url_of_the_tensorflow_python_package    
Find the one you want. Remember to choose one with GPU support. For earlier version, simply find one link and change the version number accordingly.    
    
Install Jupyter notebook:    
`(tf_1_2) $ conda install jupyter`    
    
To make Jupyter notebook aware of differen conda environments, install nb_conda:    
`(tf_1_2) $ conda install nb_conda` 
    
 Viola ! Start a jupyter notebook and start doing some deep learning !!!     
 `(tf_1_2) $ jupyter notebook`    
