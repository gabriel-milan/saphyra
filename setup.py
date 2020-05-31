import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
  name = 'saphyra',
  version = '2.0.4',
  license='GPL-3.0',
  description = '',
  long_description = long_description,
  long_description_content_type="text/markdown",
  packages=setuptools.find_packages(),
  author = 'João Victor da Fonseca Pinto',
  author_email = 'jodafons@lps.ufrj.br',
  url = 'https://github.com/jodafons/saphyra',
  keywords = ['framework', 'flexibility', 'python', 'online', 'machine learning', 'deep learning'],
  install_requires=[
          #'tensorflow', # this cause an missmatch into the tensorflow docker image. We will install manually in case of not found.
          'keras',
          'numpy',
          'six',
          'scipy',
          'future',
          'sklearn',
          'Gaugi',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)
