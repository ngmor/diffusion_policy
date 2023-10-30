export env_dir=$1

if [ -z "${env_dir}" ]
then
    echo "Error: target directory must be provided."
    exit
fi

python3 -m venv ${env_dir}
source ${env_dir}/bin/activate
pip install pip==22.2.2
pip install ipykernel==6.16
pip install jinja2
pip install typeguard
pip install pyyaml
pip install hydra-core==1.2.0
pip install dill==0.3.5.1
pip install torch==2.0.1
pip install wandb==0.13.3
pip install tqdm==4.64.1
pip install numpy==1.23.3
pip install einops==0.4.1
pip install diffusers==0.11.1
pip install zarr==2.12.0
pip install torchvision==0.15.2
pip install pandas
pip install numba==0.56.4
pip install gym==0.21.0
pip install pygame==2.1.2
pip install pymunk==6.2.1
pip install shapely==1.8.4
pip install opencv-python==4.6.0.66
pip install scikit-image==0.19.3
pip install av==10.0.0
pip install imagecodecs==2022.8.8
pip install robomimic==0.2.0
pip install cuda-python==11.6
pip install threadpoolctl==3.1.0
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.0"
pip install tensorboard==2.10.1
pip install tensorboardx==2.5.1
pip install tensorflow==2.10.1
pip install tensorflow-probability==0.18.0
pip install tf-agents==0.15.0
pip install pygame==2.1.2
pip install pybullet-svl==3.1.6.4
pip install matplotlib==3.6.1
# requires mujoco py dependencies libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
pip install free-mujoco-py==2.1.6
# Can't install with Python 3.10
# pip install robosuite@https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz
pip install pynput==1.7.6
# requires librealsense https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md
# pip install pyrealsense2==2.51.1.4348
# pip install spnav@https://github.com/cheng-chi/spnav/archive/c1c938ebe3cc542db4685e0d13850ff1abfdb943.tar.gz
pip install atomics==1.0.2

touch ${env_dir}/COLCON_IGNORE