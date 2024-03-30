from setuptools import setup, find_packages

setup(
    name='adversarial-noise-generator',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'Pillow',
        'matplotlib',
        'numpy',

        
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
        ]
    },
    python_requires='>=3.6',

    entry_points={
    'console_scripts': [
        'generate-adversarial=adversarial.generate_examples:main',
    ],
},

)
