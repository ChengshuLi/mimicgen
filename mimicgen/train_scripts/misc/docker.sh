# docker usage

# login to nvidia docker registry
# website to get keys: https://org.ngc.nvidia.com/setup/personal-keys
docker login nvcr.io

# need to have sudo, otherwise permission denied
sudo ./docker/build_doppelmaker.sh 

