#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
WORK_DIR=/mnt/wk
IMAGE_PATH=/img/h3_syspart.img

# copy U-boot in the header of the SD card
dd if=/dev/mmcblk0 of=/dev/mmcblk2 bs=512 count=2048
# create a new partition table on EMMC
# since we might have destroyed the partition table by copying U-boot
sfdisk --delete /dev/mmcblk2
echo ',,L' | sfdisk /dev/mmcblk2
mkfs.ext4 /dev/mmcblk2p1
dd if=$IMAGE_PATH of=/dev/mmcblk2p1 status=progress
mount /dev/mmcblk2p1 $WORK_DIR
cp $SCRIPT_DIR/boot.cmd $WORK_DIR/boot/
cp $SCRIPT_DIR/boot.scr $WORK_DIR/boot/
cp $SCRIPT_DIR/dietpiEnv.txt $WORK_DIR/boot/
cp $SCRIPT_DIR/fstab $WORK_DIR/etc/
echo "------------------"
echo "please unplug the SD card and reboot, and run the following commands:"
echo "\tresize2fs /dev/mmcblk2p1"
echo "------------------"
echo "disable systemd service for eth0 if you don't want to wait for the network to boot up"
echo "\tsystemctl disable ifup@eth0.service"

# some notes
# mount /dev/mmcblk2p1 /mnt/wk
# vim /mnt/wk/boot/boot.cmd
# vim /mnt/wk/boot/dietpiEnv.txt
# vim /mnt/wk/etc/fstab
# # change to /dev/mmcblk2p1
# mkimage -C none -A arm -T script -d /mnt/wk/boot/boot.cmd /mnt/wk/boot/boot.scr

# after the new reboot into the new system
# https://askubuntu.com/questions/24027/how-can-i-resize-an-ext-root-partition-at-runtime
# resize2fs /dev/mmcblk2p1

