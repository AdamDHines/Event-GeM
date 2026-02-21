sudo groupadd -f inivation && sudo usermod -aG inivation $USER

echo 'SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{idVendor}=="152a", ATTR{idProduct}=="841a", MODE:="0660", GROUP:="inivation", TAG+="uaccess"' | sudo tee /etc/udev/rules.d/99-inivation-davis.rules >/dev/null

sudo udevadm control --reload-rules && sudo udevadm trigger
