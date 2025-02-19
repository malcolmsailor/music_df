from setuptools import find_packages, setup

setup(
    name="music_df",
    version="0.0.1",
    author="Malcolm Sailor",
    author_email="malcolm.sailor@gmail.com",
    description="TODO",
    long_description="TODO",
    long_description_content_type="text/markdown",
    install_requires=["pandas", "mido", "pyyaml", "music21", "omegaconf", "mspell"],
    extras_require={"plotting": ["matplotlib"], "humdrum_export": ["metricker"]},
    url="TODO",
    project_urls={
        "Bug Tracker": "TODO",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # I think including the scripts directory is a bit of a hack
    packages=find_packages() + ["scripts"],
)
