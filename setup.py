from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'px4_offboard'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/px4_offboard/launch', glob('launch/*.py')),
        ('share/px4_offboard/vision',  glob('vision/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='thedarklord',
    maintainer_email='thedarklord@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'takeoff_hold = px4_offboard.takeoff_hold:main',
            'autonomous_land = px4_offboard.autonomous_land:main',
            'safe_land = px4_offboard.safe_land:main'
        ],
    },
)
