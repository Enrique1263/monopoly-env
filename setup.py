from setuptools import setup, find_packages

setup(
    name='monopoly_env',
    version='0.1',
    packages=find_packages(),
    install_requires=['gym', 'pygame', 'numpy'],
    include_package_data=True,
    description='A custom gym environment for Monopoly with Pygame board',
    author='Enrique',
    author_email='tu.email@example.com',
    url='https://github.com/Enrique1263/monopoly-env',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
