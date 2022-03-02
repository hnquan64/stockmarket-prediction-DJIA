from setuptools import setup

setup(
   name='python_ie221',
   version='1.0',
   description='Use SKLearn for NLP',
   author='LuatTB_QuanHN_QuanNM',
   author_email='18521068@gm.uit.edu.vn',
   packages=['python_ie221'],  #same as name
   install_requires=['matplotlib', 'scikit-learn', 'numpy','pandas', 'pytest','nltk','seaborn'], #external packages as dependencies
)