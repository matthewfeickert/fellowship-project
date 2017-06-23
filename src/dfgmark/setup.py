from setuptools import setup

setup(name='dfgmark',
      version='0.0.01',
      description='provides fit benchmarks for data flow graph frameworks',
      url='https://github.com/matthewfeickert/fellowship-project',
      author='Matthew Feickert',
      author_email='matthew.feicket@cern.ch',
      license='MIT',
      packages=find_packages(where="src"),
      package_dir={"": "src"},
      zip_safe=False)
