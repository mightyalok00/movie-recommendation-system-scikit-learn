from setuptools import setup, find_packages

setup(
    name="movie_recommendation_system",
    version="1.0.0",
    author="Alok Agarwal",
    description="Enterprise Movie Recommendation System using MovieLens 32M and Scikit-Learn",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mightyalok00/movie-recommendation-system-scikit-learn",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
        "streamlit>=1.28.0",
    ],
)
