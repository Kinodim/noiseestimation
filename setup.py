try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
        'name': 'Noise Estimation'
        'version': '0.1',
        'description': 'Adaptive measurement noise estimation for Kalman filtering',
        'author': 'Dominik Schulz',
        'packages': ['noiseestimation'],
        'install_requires': ['nose', 'numpy', 'matplotlib', 'filterpy']
}

setup(**config)
