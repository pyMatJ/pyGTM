from setuptools import setup

setup(
   name='pyGTM',
   version='2.0.0',
   description='Generalized Transfer Matrix program',
   author='Mathieu Jeannin',
   author_email='math.jeannin@free.fr',
   packages=['GTM'],
   install_requires=['numpy', 'scipy', 'matplotlib'], #external packages as dependencies
   package_dir={'GTM': 'GTM'},
   package_data={'GTM': ['GTM/MaterialData/*']},
   scripts=[
           'GTM/GTMcore.py',
           'GTM/Permittivities.py',
            ],
   include_package_data=True
)