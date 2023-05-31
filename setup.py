import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="galdyn-entropy",
    version="0.1",
    author="Leandro Beraldo e Silva",
    author_email="lberaldoesilva@gmail.com",
    description="Entropy estimators for galactic dynamics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include=['galdyn-entropy','galdyn-entropy.*']),
    python_requires='>=3',
    install_requires=["numpy","scipy"],
    package_data={"": ["README.md","LICENSE"]},
    license='MIT'
)
