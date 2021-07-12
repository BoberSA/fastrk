import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rastrk",
    version="0.0.1",
    author="Bober S.A.",
    author_email="stas.bober@gmail.com",
    description=\
        "FastRK, a generator of fast jit-compiled code for ODE propagation by ERK methods with adaptive step and events",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://bitbucket.org/stas_bober/orbipy",
    packages=setuptools.find_packages(),
    # package_data={
    #         'fastrk': ['_events'],
    # },
    license='MIT',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy~=1.19.2',
                      'numba~=0.51.2',
                      'sympy~=1.6.2',
                      ],
    python_requires='>=3.8',
)