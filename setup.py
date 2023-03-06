from setuptools import setup, find_packages

setup(name='ChatGPT API',
      version='0.1',
      description='Gpt API',
      author='Coder Sun',
      packages=find_packages(),
      install_requires=[
          'openai==0.2.0',
          'package2'
      ],
      entry_points={
          'console_scripts': ['mycli=mymodule:cli'],
      }
      )