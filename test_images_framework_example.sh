echo 'Using Docker to start the container and run tests ...'
sudo docker build --force-rm --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)" -t images_framework_image .
sudo docker volume create --name images_framework_volume
sudo docker run --name images_framework_container -v images_framework_volume:/home/username --rm -it -d images_framework_image bash
sudo docker exec -w /home/username/ images_framework_container python images_framework/test/images_framework_test.py
sudo docker stop images_framework_container
sudo docker image prune
