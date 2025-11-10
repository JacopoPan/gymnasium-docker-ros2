from setuptools import find_packages, setup

package_name = 'dynamics'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='JacopoPan',
    maintainer_email='jacopo.pan@gmail.com',
    description='Dynamics package',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dynamics = dynamics.dynamics_node:main',
        ],
    },
)
