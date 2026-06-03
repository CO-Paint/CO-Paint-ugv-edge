from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'arduino_mecanum_bridge'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ilju',
    maintainer_email='iljujjang@gmail.com',
    description='ROS2 serial bridge for Arduino Mega mecanum motor controller',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'serial_bridge = arduino_mecanum_bridge.serial_bridge:main',
        ],
    },
)
