from setuptools import setup, find_packages


setup(
    name="despeckling",
    version="0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': {
            'aef = methods.aef.run:main',
            'bm3d = methods.bm3d.run:main',
            'fans = methods.fans.run:main',
            'gbf = methods.gbf.run:main',
            'gnlm = methods.gnlm.run:main',
            'merlin = methods.merlin.run:main',
            'ppb = methods.ppb.run:main',
            'sar2sar = methods.sar2sar.run:main',
            'sarbm3d = methods.sarbm3d.run:main',
            'speckle2void = methods.speckle2void.run:main',
            'tbog = methods.tbog.run:main',
        }
    },
)
