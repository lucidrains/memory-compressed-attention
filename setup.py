from setuptools import setup, find_packages

setup(
  name = 'memory_compressed_attention',
  packages = find_packages(),
  version = '0.0.4',
  license='MIT',
  description = 'Memory-Compressed Self Attention',
  long_description_content_type = 'text/markdown',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/memory-compressed-attention',
  keywords = ['transformers', 'artificial intelligence', 'attention mechanism'],
  install_requires=[
    'torch'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
