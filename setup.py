"""
Setup script for the Floor Monitoring Application.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="floor-monitoring-app",
    version="1.0.0",
    author="Floor Monitoring Team",
    author_email="team@floormonitoring.com",
    description="A surveillance system for monitoring worker presence on factory floors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/floor-monitoring-app",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Manufacturing",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "floor-monitor=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["../config/*.json", "../assets/*"],
    },
)