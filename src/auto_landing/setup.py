from setuptools import find_packages, setup

package_name = 'auto_landing'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ugv',
    maintainer_email='ugv@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pid_landing_node = auto_landing.pid_landing_node:main',
            'pid_align_landing_node = auto_landing.pid_align_landing_node:main'
        ],
    },
)
