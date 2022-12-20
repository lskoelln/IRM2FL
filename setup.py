from setuptools import setup, find_packages

setup(name='irm2fl',
      description='IRM2FL - Neural Network Frameworks for Artificial Labelling in IRM',
      author='Lisa Kölln',
      version='0.1.0',
      license='BSD 3-Clause License',
      packages = find_packages(),
      python_requires='==3.9',
      install_requires=[
          "tensorflow==2.9",
          "tensorflow-addons<=0.17",
          "six",
          "tqdm",
          "numpy",
          "matplotlib",
          "tifffile",
          "scipy",
          "jupyter",
          "augmend @ git+https://github.com/stardist/augmend.git#egg=augmend-0.1.0",
      ],
      dependency_links = [
          "git+https://github.com/stardist/augmend.git#egg=augmend-0.1.0"
      ]
      )